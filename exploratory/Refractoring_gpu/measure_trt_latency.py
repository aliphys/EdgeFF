"""
measure_trt_latency.py
=======================

Download trained models from W&B, compile to Torch-TensorRT, and
measure latency and energy on Jetson devices.

This script follows the evaluation flow in `eval.py` but compiles the
model with Torch-TensorRT and measures inference latency for a sweep of
batch sizes (1..512 by default). It also uses the project's hardware
monitor (if available) to collect energy metrics.

Usage:
    python measure_trt_latency.py --config eval_config.yaml

The config format is the same as `eval.py` (project, sweep_id, dataset,
inference_batch_sizes, hw_interval_ms).
"""

import argparse
import os
from pathlib import Path
import yaml
import time
import sys
import socket

import torch
import wandb
import types
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import shutil

import seaborn as sns
import numpy as np

try:
    import torch_tensorrt as trt
except Exception:
    trt = None


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_hw_monitor(interval_ms=500):
    try:
        from tegrats_monitor import TegratsMonitor
        hw = TegratsMonitor(interval_ms=interval_ms)
        hw.start()
        return hw
    except Exception:
        return None


class TRTWrapper(torch.nn.Module):
    """Wrap the original model to expose a `forward(x)` suitable for TRT.

    The underlying model exposes `predict_one_pass(x, batch_size=...)`.
    The wrapper forwards a tensor batch to that method and returns the
    class predictions as a tensor.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Implement the inference forward here to avoid calling functions
        # that use `clone()` (which Torch-TensorRT's dynamo may not support).
        # We'll perform the "neutral" overlay without clone and propagate
        # through the layers to produce the softmax probabilities (float).
        x = x.to(self.model.device)

        # Neutral overlay without clone: build new tensor using concatenation
        # to avoid in-place indexing / scatter ops that TRT conversion dislikes.
        batch = x.shape[0]
        dtype = x.dtype
        device = x.device
        if self.model.is_color:
            pixels_per_channel = x.shape[1] // 3
            # Red channel: replace first 10 pixels with 0.1, keep rest
            r_prefix = torch.full((batch, 10), 0.1, dtype=dtype, device=device)
            r_suffix = x[:, 10:pixels_per_channel]
            r = torch.cat((r_prefix, r_suffix), dim=1)

            # Green channel
            g_prefix = torch.full((batch, 10), 0.1, dtype=dtype, device=device)
            g_suffix = x[:, pixels_per_channel + 10:2 * pixels_per_channel]
            g = torch.cat((g_prefix, g_suffix), dim=1)

            # Blue channel
            b_prefix = torch.full((batch, 10), 0.1, dtype=dtype, device=device)
            b_suffix = x[:, 2 * pixels_per_channel + 10:3 * pixels_per_channel]
            b = torch.cat((b_prefix, b_suffix), dim=1)

            x2 = torch.cat((r, g, b), dim=1)
        else:
            prefix = torch.full((batch, 10), 0.1, dtype=dtype, device=device)
            suffix = x[:, 10:]
            x2 = torch.cat((prefix, suffix), dim=1)

        h = x2
        softmax_layer_input = None
        # propagate through layers and build softmax input cumulatively
        for layer, softmax_layer in zip(self.model.layers, self.model.softmax_layers):
            h = layer(h)
            if softmax_layer_input is None:
                softmax_layer_input = h
            else:
                softmax_layer_input = torch.cat((softmax_layer_input, h), 1)

        # get logits and softmax probabilities from the last softmax layer
        logits, probs = softmax_layer(softmax_layer_input)
        return probs


def compile_tensorrt(model, max_batch=512, use_fp16=True, input_dim=None):
    if trt is None:
        raise RuntimeError("torch_tensorrt is not available")

    wrapper = TRTWrapper(model)
    wrapper.eval()

    if input_dim is None:
        # attempt to infer input dim from first layer
        try:
            input_dim = model.layers[0].in_features
        except Exception:
            raise RuntimeError("Could not infer model input dimension; pass input_dim")

    # Choose input dtype for the Input descriptor (engine precision is separate)
    input_dtype = torch.float16 if use_fp16 else torch.float32

    # dynamic batch dimension: specify min/opt/max only (do NOT pass 'shape')
    trt_input = trt.Input(
        min_shape=[1, input_dim],
        opt_shape=[8, input_dim],
        max_shape=[max_batch, input_dim],
        dtype=input_dtype,
    )

    enabled_precisions = {torch.float32}
    if use_fp16:
        enabled_precisions.add(torch.float16)

    # Try compiling; allow exceptions to bubble up so caller can fallback
    trt_mod = trt.compile(
        wrapper,
        inputs=[trt_input],
        enabled_precisions=enabled_precisions,
        workspace_size=1 << 28,
    )

    return trt_mod, input_dtype


def export_model_to_onnx(wrapper, onnx_path, input_dim, input_dtype, device):
    # wrapper: module with forward(x) ready for inference
    wrapper.eval()
    dummy = torch.randn((1, input_dim), device=device)
    if input_dtype == torch.float16:
        dummy = dummy.half()
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")


def run_trtexec_latency(onnx_path, fp16=False, max_batch=512, avg_runs=100, workspace=2048):
    # Build trtexec command and parse its stdout for latency numbers.
    engine_path = onnx_path + '.engine'
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        '--explicitBatch',
        f'--saveEngine={engine_path}',
        f'--workspace={workspace}',
        f'--avgRuns={avg_runs}',
    ]
    if fp16:
        cmd.append('--fp16')

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        out = proc.stdout
    except subprocess.CalledProcessError as e:
        out = e.stdout or ''
        raise RuntimeError(f"trtexec failed: {e}\nOutput:\n{out}")

    # Parse output for latency. Look for common patterns like 'Average :' or 'mean'
    import re
    ms = None
    # common trtexec prints lines like 'mean: 0.123 ms' or 'Average over ...: 0.123 ms'
    patterns = [r"mean[^\d]*(\d+\.\d+)\s*ms", r"average[^\d]*(\d+\.\d+)\s*ms", r"Avg.*?(\d+\.\d+)\s*ms", r"GPU.*?(\d+\.\d+)\s*ms"]
    for p in patterns:
        m = re.search(p, out, re.IGNORECASE)
        if m:
            ms = float(m.group(1))
            break

    if ms is None:
        # fallback: try find all floats followed by ms and take last
        m_all = re.findall(r"(\d+\.\d+)\s*ms", out)
        if m_all:
            ms = float(m_all[-1])

    return {'trtexec_stdout': out, 'latency_ms': ms, 'engine_path': engine_path}


def measure_latency(trt_model, test_inputs, batch_size, device, input_dtype=torch.float32, hw_monitor=None, warmup=50):
    """Measure latency and return aggregated metrics plus per-iteration timings.

    Returns a dict containing:
      - latency_ms (which is the true latency measured at batch size 1)
      - latency_per_batch_ms (time to process the given batch_size)
      - memory_mb
      - hw_metrics
      - per_iter_batch_ms: list of measured batch latencies (ms) for each iteration
    """
    # Ensure dtype matches engine expectation
    if input_dtype == torch.float16:
        test_inputs = test_inputs.half()
    else:
        test_inputs = test_inputs.float()

    num_samples = test_inputs.shape[0]

    # Warm-up (not timed): 3 full passes over the test set
    torch.cuda.synchronize()
    for _ in range(3):
        for i in range(0, num_samples, batch_size):
            batch = test_inputs[i:i + batch_size].contiguous()
            _ = trt_model(batch)
    torch.cuda.synchronize()

    # If hw monitor available, start measurement
    if hw_monitor:
        hw_monitor.start_inference_measurement()

    per_iter_ms = []
    # Measure per-iteration over the entire dataset using CUDA events
    for i in range(0, num_samples, batch_size):
        batch = test_inputs[i:i + batch_size].contiguous()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        _ = trt_model(batch)
        end_evt.record()
        # synchronize to make sure the events are recorded
        torch.cuda.synchronize()
        per_iter_ms.append(start_evt.elapsed_time(end_evt))

    total_ms = sum(per_iter_ms)

    if hw_monitor:
        batch_metrics = hw_monitor.stop_inference_measurement(num_samples)
    else:
        batch_metrics = None

    num_batches = len(per_iter_ms)
    latency_per_batch_ms = total_ms / num_batches if num_batches else 0

    memory_mb = torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0

    metrics = {
        'latency_ms': latency_per_batch_ms, 
        'memory_mb': memory_mb,
        'hw_metrics': batch_metrics,
        'per_iter_batch_ms': per_iter_ms,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='YAML config (same as eval.py)')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 for TRT compile')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--max_batch', type=int, default=512)
    parser.add_argument('--socket-timeout', type=int, default=60, help='Socket timeout in seconds for network ops')
    args = parser.parse_args()

    # set global socket timeout to reduce socket.send() timeout errors
    try:
        socket.setdefaulttimeout(args.socket_timeout)
    except Exception:
        pass

    config = load_config(args.config)
    # Initialize wandb safely — network/socket errors can occur on upload/download.
    wandb_online = True

    # helper for retrying network calls with exponential backoff
    def retry_call(fn, retries=3, backoff_sec=5, *fargs, **fkwargs):
        last_exc = None
        for i in range(retries):
            try:
                return fn(*fargs, **fkwargs)
            except Exception as e:
                last_exc = e
                sleep = backoff_sec * (2 ** i)
                print(f"Warning: call {getattr(fn, '__name__', str(fn))} failed (attempt {i+1}/{retries}): {e}. Retrying in {sleep}s")
                time.sleep(sleep)
        raise last_exc

    try:
        # retry wandb.init in case of transient network errors
        try:
            retry_call(wandb.init, retries=3, backoff_sec=5, project=config.get('project', 'edgeff-network-width'), job_type='trt-eval', config=config)
        except Exception as e:
            print("Warning: wandb.init() failed after retries. Continuing with wandb disabled:", e)
            wandb_online = False
    except Exception:
        wandb_online = False

    try:
        api = retry_call(lambda: wandb.Api(), retries=3, backoff_sec=5) if wandb_online else None
    except Exception as e:
        print("Warning: wandb.Api() failed after retries. Can't query sweep runs:", e)
        api = None
        wandb_online = False
    project_name = config.get('project')
    sweep_id = config.get('sweep_id')
    if not sweep_id:
        raise ValueError('sweep_id required in config')

    if api is None:
        raise RuntimeError("Cannot access W&B API to fetch sweep runs. Check network or set WANDB_MODE=offline.")
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    runs = sweep.runs

    # Hardware monitor
    hw_monitor = get_hw_monitor(interval_ms=config.get('hw_interval_ms', 500))

    # Dataset loaders from eval.py style
    from eval import dataset_loaders
    dataset_name = config.get('dataset', 'MNIST')
    batch_sizes = config.get('inference_batch_sizes', list(range(1, args.max_batch + 1)))

    test_loader, onehot_max_value, is_color = dataset_loaders(dataset_name, test_batch_size=512)
    test_inputs = torch.cat([d for d, _ in test_loader], dim=0)
    test_targets = torch.cat([t for _, t in test_loader], dim=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    # global, monotonically increasing step for wandb logs across all runs
    log_step = 0
    # accumulate measurements across runs for global plots (e.g., width vs latency)
    global_measurements = []
    for run in runs:
        run_id = run.id
        print(f"Processing run {run_id}")
        artifacts = list(run.logged_artifacts())
        model_artifact = None
        for art in artifacts:
            if art.type == 'model':
                model_artifact = art
                break
        if model_artifact is None:
            print(f"No model artifact for run {run_id}, skipping")
            continue

        # try to read metadata (e.g., width) for better plotting in W&B
        try:
            width = model_artifact.metadata.get('width', None) if model_artifact.metadata else None
        except Exception:
            width = None

        try:
            artifact_dir = retry_call(model_artifact.download, retries=4, backoff_sec=5)
        except Exception as e:
            print(f"Error: failed to download artifact for run {run_id}: {e}")
            if wandb_online:
                print("Stopping because artifact download failed")
                sys.exit(1)
            else:
                continue
        model_path = os.path.join(artifact_dir, 'temp_')
        device_map = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_path, map_location=device_map, weights_only=False)

        # Ensure model on device
        model.device = device
        for layer in model.layers:
            layer.to(device)
        for s in model.softmax_layers:
            s.to(device)
        model.onehot_max_value = onehot_max_value
        model.is_color = is_color

        # PyTorch convention: Module.train(self, mode=True) is used to set
        # training/eval mode. In this codebase several classes (Net,
        # SoftmaxLayer, Layer) define a custom `train(...)` with different
        # signatures which breaks tooling (Torch-TensorRT, torch.jit, etc.)
        # because they call `module.train()` without args. To avoid that
        # problem we monkey-patch all modules' `train` to the standard
        # torch.nn.Module.train bound method before compiling.
        def _make_train_override(m):
            def _train(self, mode=True):
                return torch.nn.Module.train(self, mode)
            return types.MethodType(_train, m)

        for m in model.modules():
            try:
                m.train = _make_train_override(m)
            except Exception:
                # be conservative: if we can't override, continue
                pass

        # infer input dim
        input_dim = model.layers[0].in_features

        # compile to TRT (single compiled module used for all batch sizes via dynamic shape)
        print("Compiling model to Torch-TensorRT (this may take a while)")
        onnx_fallback_used = False
        trt_mod = None
        trt_input_dtype = torch.float32
        trt_compile_success = False
        try:
            trt_mod, trt_input_dtype = compile_tensorrt(model, max_batch=args.max_batch, use_fp16=args.fp16, input_dim=input_dim)
            trt_compile_success = True
        except Exception as e:
            print("Torch-TensorRT compile failed:", e)
            # Try ONNX -> trtexec fallback
            print("Attempting ONNX fallback using trtexec...")
            onnx_fallback_used = True
            onnx_path = os.path.join(artifact_dir, f'model_{run_id}.onnx')
            try:
                export_model_to_onnx(TRTWrapper(model), onnx_path, input_dim, torch.float16 if args.fp16 else torch.float32, device)
                trtexec_result = run_trtexec_latency(onnx_path, fp16=args.fp16, max_batch=args.max_batch, avg_runs=args.iters)
                print("trtexec result:", trtexec_result['latency_ms'], "ms")
            except Exception as e2:
                print("ONNX fallback also failed:", e2)
                print("Stopping evaluation due to TRT/ONNX compile failure.")
                sys.exit(1)

        # Save compiled TRT module (as TorchScript) so it can be reused and uploaded
        try:
            save_name = f'trt_model_{run_id}.pt'
            save_path = os.path.join(artifact_dir, save_name)
            # Try scripting and saving the compiled module
            try:
                scripted = torch.jit.script(trt_mod)
            except Exception:
                # Fallback to tracing with a dummy input (match dtype)
                dummy = torch.randn((1, input_dim), device=device)
                if trt_input_dtype == torch.float16:
                    dummy = dummy.half()
                scripted = torch.jit.trace(trt_mod, dummy)
            torch.jit.save(scripted, save_path)

            # Upload the saved engine as a W&B artifact
            try:
                art_name = f"trt-model-{run_id}"
                artifact = wandb.Artifact(art_name, type='trt_model')
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)
            except Exception as e:
                print("Warning: failed to upload TRT artifact to W&B:", e)
        except Exception as e:
            print("Warning: failed to save TRT module to disk:", e)

        # evaluate across batch sizes
        measurements = []
        precision = 'fp16' if args.fp16 else 'fp32'
        for bs in batch_sizes:
            if bs > args.max_batch:
                continue
            print(f"Measuring batch_size={bs}")

            if not onnx_fallback_used:
                # Warm-up & measure using compiled TRT module over full test set
                metrics = measure_latency(trt_mod, test_inputs, bs, device, input_dtype=trt_input_dtype, hw_monitor=hw_monitor, warmup=args.warmup)

                # If hardware metrics exist, compute per-sample energy
                hw = metrics.get('hw_metrics')
                energy_per_sample_mj = None
                avg_power_mw = None
                if hw:
                    energy_per_sample_mj = hw.get('inference/energy_per_sample_mj')
                    avg_power_mw = hw.get('inference/avg_power_during_inference_mw')

                latency_ms = metrics['latency_ms']
                # per-iteration batch latencies (ms)
                per_iter_batch_ms = metrics.get('per_iter_batch_ms')
                per_iter_sample_ms = None
                if per_iter_batch_ms:
                    per_iter_sample_ms = per_iter_batch_ms
            else:
                # trtexec result from earlier ONNX fallback
                # trtexec reports latency per batch — treat as batch latency
                try:
                    # run_trtexec_latency returns latency in ms for avgRuns; use that value
                    trtexec_out = trtexec_result
                    latency_ms = trtexec_out.get('latency_ms') or 0
                    energy_per_sample_mj = None
                    avg_power_mw = None
                    # If hw_monitor available, measure energy using the original PyTorch model
                    if hw_monitor:
                        from Evaluation import eval_with_inference_measurement
                        inference_metrics = eval_with_inference_measurement(model, test_inputs, test_targets, power_monitor=hw_monitor, set_name='test', batch_size=bs)
                        energy_per_sample_mj = inference_metrics.get('test/inference_energy_per_sample_mj')
                        avg_power_mw = inference_metrics.get('test/inference_avg_power_mw')
                except Exception as e:
                    print('Warning: ONNX trtexec measurement handling failed:', e)
                    latency_ms = None
                    latency_per_batch_ms = 0
                    per_iter_sample_ms = None

            # record measurement locally for aggregation
            throughput_samples_per_sec = (bs / (latency_ms / 1000.0)) if latency_ms else 0
            measurements.append({
                'batch_size': bs,
                'latency_ms': latency_ms,
                'throughput_samples_per_sec': throughput_samples_per_sec,
                'memory_mb': torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0,
                'energy_per_sample_mj': energy_per_sample_mj or 0,
                'avg_power_mw': avg_power_mw or 0,
                'per_iter_sample_ms': per_iter_sample_ms,
            })

            # add to global accumulator for cross-run boxplots (grouped by width)
            global_measurements.append({
                'width': width, 
                'batch_size': bs, 
                'latency_ms': latency_ms,
                'throughput_samples_per_sec': throughput_samples_per_sec,
                'energy_per_sample_mj': energy_per_sample_mj,
                'avg_power_mw': avg_power_mw,
                'memory_mb': torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0,
                'per_iter_sample_ms': per_iter_sample_ms
            })

            # Log to wandb (safe wrapper) with richer metadata and step
            def safe_wandb_log(payload, step=None):
                """Log to wandb without sending an explicit step to avoid
                non-monotonic step warnings. WandB will auto-increment steps.
                The `step` parameter is accepted for backward compatibility but
                ignored.
                """
                if not wandb_online:
                    print("wandb disabled — skipping log")
                    return
                try:
                    # Do NOT pass `step=` to wandb.log to avoid out-of-order
                    # step warnings when external code has already advanced
                    # the internal step counter. Let wandb auto-increment.
                    retry_call(wandb.log, retries=3, backoff_sec=3, data=payload)
                except Exception as e:
                    print("Warning: wandb.log failed after retries (network/socket). Continuing:", e)

            payload = {
                'run_id': run_id,
                'width': width,
                'batch_size': bs,
                f'throughput_vs_batch/{run_id}': throughput_samples_per_sec,
                f'latency_vs_batch/{run_id}': latency_ms,
                'latency_ms': latency_ms,
                'throughput_samples_per_sec': throughput_samples_per_sec,
                'memory_mb': torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0,
                'energy_per_sample_mj': energy_per_sample_mj or 0,
                'avg_power_mw': avg_power_mw or 0,
                'trt_compile_success': trt_compile_success,
                'onnx_fallback_used': onnx_fallback_used,
                'precision': precision,
                'warmup_iters': args.warmup,
                'measured_iters': args.iters,
                'device': str(device),
            }

            safe_wandb_log(payload, step=log_step)
            log_step += 1

        # after sweeping batch sizes, compute summary stats and log them
        if measurements:
            best = max(measurements, key=lambda m: m['throughput_samples_per_sec'])
            latency_at_1 = next((m['latency_ms'] for m in measurements if m['batch_size'] == 1), None)
            summary = {
                'run_id': run_id,
                'best_throughput_samples_per_sec': best['throughput_samples_per_sec'],
                'best_throughput_batch_size': best['batch_size'],
                'latency_at_batch1_ms': latency_at_1,
                'trt_compile_success': trt_compile_success,
                'onnx_fallback_used': onnx_fallback_used,
                'precision': precision,
            }
            # Create automatic visualization panels (images + table) and log them to W&B.
            def _create_and_log_charts(measurements_list, run_id, width, safe_log_fn):
                # Convert to DataFrame
                try:
                    df = pd.DataFrame(measurements_list)
                except Exception:
                    return

                tmpdir = tempfile.mkdtemp(prefix=f"wandb_plots_{run_id}_")
                try:
                    plots = {}
                    # Throughput vs Batch Size
                    try:
                        fig, ax = plt.subplots()
                        ax.plot(df['batch_size'], df['throughput_samples_per_sec'], marker='o')
                        ax.set_xlabel('batch_size')
                        ax.set_ylabel('throughput (samples/sec)')
                        ax.set_title(f'Throughput vs Batch Size (Width: {width})')
                        path = os.path.join(tmpdir, 'throughput_vs_batch.png')
                        fig.savefig(path, bbox_inches='tight')
                        plt.close(fig)
                        plots[f'per_model_plots/throughput_width_{width}'] = path
                    except Exception:
                        pass

                    # Latency ridge plot by Batch Size
                    try:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        hist_data = []
                        labels = []
                        for m in measurements_list:
                            samples = m.get('per_iter_sample_ms')
                            if not samples:
                                samples = m.get('per_iter_batch_ms')
                            
                            if samples:
                                hist_data.append(samples)
                                labels.append(f"Batch {m['batch_size']}")
                            else:
                                val = m.get('latency_ms')
                                if val is not None:
                                    hist_data.append([val])
                                    labels.append(f"Batch {m['batch_size']}")

                        if hist_data:
                            import numpy as np
                            # create a ridge plot (joyplot) using shifted histograms
                            bins = np.linspace(0, 3, 100)
                            overlap = 0.5
                            max_height = 0
                            
                            # calculate max histogram count for scaling shifts
                            hists = [np.histogram(data, bins=bins)[0] for data in hist_data]
                            if hists:
                                max_height = max([h.max() for h in hists])
                            
                            shift = max_height * overlap if max_height > 0 else 1
                            
                            for i, (data, label) in enumerate(zip(hist_data, labels)):
                                counts, _ = np.histogram(data, bins=bins)
                                base = i * shift
                                ax.fill_between(bins[:-1], base, base + counts, alpha=0.7, label=label)
                                ax.plot(bins[:-1], base + counts, color='black', lw=0.5)
                            
                            ax.set_xlabel('Latency (ms)')
                            ax.set_ylabel('Frequency (Shifted)')
                            # logarithmic scale might look weird with shifted bases, so we disable it
                            ax.set_xlim(0, 3)
                            
                            # custom y-ticks to show labels
                            ax.set_yticks([i * shift for i in range(len(labels))])
                            ax.set_yticklabels(labels)
                            
                            ax.set_title(f'Latency Distribution by Batch Size (Width: {width})')
                            path = os.path.join(tmpdir, 'latency_hist_by_batch.png')
                            fig.tight_layout()
                            fig.savefig(path, bbox_inches='tight')
                            plt.close(fig)
                            plots[f'per_model_plots/latency_histogram_width_{width}'] = path
                    except Exception as e:
                        pass

                    # Energy histogram by Batch Size (if present)
                    if 'energy_per_sample_mj' in df.columns and df['energy_per_sample_mj'].any():
                        try:
                            # Group hw monitor energy events instead of line plot. Currently energy_per_sample_mj is scalar per batch size in measure_latency.
                            # Plotting single scalar points as histograms doesn't yield distributions unless multiple iterations/samples are available.
                            # We'll plot a bar chart comparing energy across batch sizes for this specific model width as a fallback, 
                            # or histogram if future changes add distribution energy data.
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.bar([str(bs) for bs in df['batch_size']], df['energy_per_sample_mj'], color='orange', alpha=0.7)
                            ax.set_xlabel('Batch Size')
                            ax.set_ylabel('Energy per sample (mJ)')
                            ax.set_title(f'Energy per Sample Distribution by Batch Size (Width: {width})')
                            path = os.path.join(tmpdir, 'energy_hist_by_batch.png')
                            fig.savefig(path, bbox_inches='tight')
                            plt.close(fig)
                            plots[f'per_model_plots/energy_histogram_width_{width}'] = path
                        except Exception:
                            pass

                    # Log plots and the measurements table
                    if safe_log_fn is not None:
                        try:
                            # log images
                            for k, p in plots.items():
                                safe_log_fn({k: wandb.Image(p)})
                            # log a table for interactive plotting in W&B
                            safe_log_fn({f'measurements_table_{run_id}': wandb.Table(dataframe=df)})
                        except Exception as e:
                            print('Warning: failed to log generated plots/tables to wandb:', e)
                finally:
                    try:
                        shutil.rmtree(tmpdir)
                    except Exception:
                        pass

            # call chart creator which logs images + table to W&B (if enabled)
            _create_and_log_charts(measurements, run_id, width, safe_wandb_log)

            # finally log the summary
            safe_wandb_log(summary)

    if hw_monitor:
        hw_monitor.stop()

    # After all runs, create summary plots comparing sizes and batch sizes
    try:
        if global_measurements:
            df_global = pd.DataFrame(global_measurements)
            df_global = df_global.dropna(subset=['width', 'batch_size'])
            
            tmpdir = tempfile.mkdtemp(prefix="wandb_global_plots_")
            try:
                plots_to_upload = {}
                
                # Combined Throughput vs Batch Size Line Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=df_global, x='batch_size', y='throughput_samples_per_sec', hue='width', marker='o', ax=ax)
                ax.set_title("Throughput vs Batch Size (All Models)")
                ax.set_xlabel("Batch Size")
                ax.set_ylabel("Throughput (samples/sec)")
                path_comb_through = os.path.join(tmpdir, 'combined_throughput.png')
                fig.savefig(path_comb_through, bbox_inches='tight')
                plt.close(fig)
                plots_to_upload['global/combined_throughput_vs_batch'] = wandb.Image(path_comb_through)

                # Combined Energy vs Batch Size Line Plot
                if 'energy_per_sample_mj' in df_global.columns and not df_global['energy_per_sample_mj'].isna().all():
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(data=df_global, x='batch_size', y='energy_per_sample_mj', hue='width', marker='o', ax=ax)
                    ax.set_title("Energy per Sample vs Batch Size (All Models)")
                    ax.set_xlabel("Batch Size")
                    ax.set_ylabel("Energy per sample (mJ)")
                    path_comb_energy = os.path.join(tmpdir, 'combined_energy.png')
                    fig.savefig(path_comb_energy, bbox_inches='tight')
                    plt.close(fig)
                    plots_to_upload['global/combined_energy_vs_batch'] = wandb.Image(path_comb_energy)

                # 1. Throughput Heatmap (Width x Batch Size)
                fig, ax = plt.subplots(figsize=(10, 8))
                pivot_throughput = df_global.pivot_table(index='width', columns='batch_size', values='throughput_samples_per_sec', aggfunc='mean')
                sns.heatmap(pivot_throughput, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
                ax.set_title("Throughput (samples/sec) by Model Width and Batch Size")
                ax.set_xlabel("Batch Size")
                ax.set_ylabel("Network Width")
                path_through_heat = os.path.join(tmpdir, 'throughput_heatmap.png')
                fig.savefig(path_through_heat, bbox_inches='tight')
                plt.close(fig)
                plots_to_upload['global/throughput_heatmap'] = wandb.Image(path_through_heat)

                # Ridge plot for Latency, Batch Size = 32
                try:
                    df_bs32 = [m for m in global_measurements if m.get('batch_size') == 32]
                    if df_bs32:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        hist_data_lat = []
                        labels_lat = []
                        
                        # Sort by width numerically
                        def get_width_val(x):
                            w = x.get('width')
                            try:
                                return float(w)
                            except (ValueError, TypeError):
                                return 0.0
                        df_bs32.sort(key=get_width_val)
                        
                        for m in df_bs32:
                            w = m.get('width')
                            samples = m.get('per_iter_sample_ms')
                            if not samples:
                                samples = m.get('per_iter_batch_ms')
                            
                            if samples:
                                hist_data_lat.append(samples)
                                labels_lat.append(f"Width {w}")
                            else:
                                val = m.get('latency_ms')
                                if val is not None:
                                    hist_data_lat.append([val])
                                    labels_lat.append(f"Width {w}")
                        
                        if hist_data_lat:
                            import numpy as np
                            bins = np.linspace(0, 3, 100)
                            overlap = 0.5
                            max_height = 0
                            hists = [np.histogram(data, bins=bins)[0] for data in hist_data_lat]
                            if hists:
                                max_height = max([h.max() for h in hists])
                            shift = max_height * overlap if max_height > 0 else 1
                            
                            for i, (data, label) in enumerate(zip(hist_data_lat, labels_lat)):
                                counts, _ = np.histogram(data, bins=bins)
                                base = i * shift
                                ax.fill_between(bins[:-1], base, base + counts, alpha=0.7, label=label)
                                ax.plot(bins[:-1], base + counts, color='black', lw=0.5)
                            
                            ax.set_xlabel('Latency (ms)')
                            ax.set_ylabel('Network Size')
                            ax.set_xlim(0, 3)
                            
                            ax.set_yticks([i * shift for i in range(len(labels_lat))])
                            ax.set_yticklabels(labels_lat)
                            ax.set_title('Latency Ridge Plot (Batch Size: 32)')
                            
                            path_lat_ridge = os.path.join(tmpdir, 'latency_ridge_bs32.png')
                            fig.tight_layout()
                            fig.savefig(path_lat_ridge, bbox_inches='tight')
                            plt.close(fig)
                            plots_to_upload['global/latency_ridge_bs32'] = wandb.Image(path_lat_ridge)
                except Exception as e:
                    pass

                # Ridge plot for Energy, Batch Size = 32
                try:
                    df_bs32 = [m for m in global_measurements if m.get('batch_size') == 32]
                    if df_bs32:
                        hist_data_eng = []
                        labels_eng = []
                        
                        # Sort by width numerically
                        def get_width_val(x):
                            w = x.get('width')
                            try:
                                return float(w)
                            except (ValueError, TypeError):
                                return 0.0
                        df_bs32.sort(key=get_width_val)
                        
                        for m in df_bs32:
                            w = m.get('width')
                            eng = m.get('energy_per_sample_mj')
                            if eng is not None:
                                hist_data_eng.append([eng])
                                labels_eng.append(f"Width {w}")
                        
                        if hist_data_eng:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            import numpy as np
                            all_eng = [e for sublist in hist_data_eng for e in sublist]
                            min_e, max_e = (0, max(all_eng) * 1.5) if all_eng else (0, 1)
                            if max_e == 0: max_e = 1
                            bins = np.linspace(min_e, max_e, 100)
                            overlap = 0.5
                            max_height = 0
                            hists = [np.histogram(data, bins=bins)[0] for data in hist_data_eng]
                            if hists:
                                max_height = max([h.max() for h in hists])
                            shift = max_height * overlap if max_height > 0 else 1
                            
                            for i, (data, label) in enumerate(zip(hist_data_eng, labels_eng)):
                                counts, _ = np.histogram(data, bins=bins)
                                base = i * shift
                                ax.fill_between(bins[:-1], base, base + counts, alpha=0.7, label=label)
                                ax.plot(bins[:-1], base + counts, color='black', lw=0.5)
                            
                            ax.set_xlabel('Energy per Sample (mJ)')
                            ax.set_ylabel('Network Size')
                            ax.set_xlim(min_e, max_e)
                            
                            ax.set_yticks([i * shift for i in range(len(labels_eng))])
                            ax.set_yticklabels(labels_eng)
                            ax.set_title('Energy Ridge Plot (Batch Size: 32)')
                            
                            path_eng_ridge = os.path.join(tmpdir, 'energy_ridge_bs32.png')
                            fig.tight_layout()
                            fig.savefig(path_eng_ridge, bbox_inches='tight')
                            plt.close(fig)
                            plots_to_upload['global/energy_ridge_bs32'] = wandb.Image(path_eng_ridge)
                except Exception as e:
                    pass

                # 2. Latency Histogram (Width)
                # group per-iteration samples by width
                by_width = {}
                for rec in global_measurements:
                    w = rec.get('width')
                    if w is None:
                        continue
                    samples = rec.get('per_iter_sample_ms')
                    if samples:
                         by_width.setdefault(w, []).extend(samples)
                    else:
                         val = rec.get('latency_ms')
                         if val is not None:
                             by_width.setdefault(w, []).append(val)

                if by_width:
                    widths = sorted(by_width.keys(), key=lambda x: (str(x)))
                    hist_data = [by_width[w] for w in widths]
                    labels = [f"Width {w}" for w in widths]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(hist_data, bins=30, label=labels, alpha=0.7)
                    ax.set_xlabel('Latency (ms)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Latency Distribution by Network Width')
                    ax.legend()
                    path_lat_hist = os.path.join(tmpdir, 'latency_hist_by_width.png')
                    fig.tight_layout()
                    fig.savefig(path_lat_hist, bbox_inches='tight')
                    plt.close(fig)
                    plots_to_upload['global/latency_by_width_histogram'] = wandb.Image(path_lat_hist)

                # 3. Energy Heatmap & Energy Histogram
                if 'energy_per_sample_mj' in df_global.columns and not df_global['energy_per_sample_mj'].isna().all():
                    # 3A. Energy Heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    pivot_energy = df_global.pivot_table(index='width', columns='batch_size', values='energy_per_sample_mj', aggfunc='mean')
                    sns.heatmap(pivot_energy, annot=True, fmt=".2f", cmap="OrRd", ax=ax)
                    ax.set_title("Energy per Sample (mJ) by Model Width and Batch Size")
                    ax.set_xlabel("Batch Size")
                    ax.set_ylabel("Network Width")
                    path_energy = os.path.join(tmpdir, 'energy_heatmap.png')
                    fig.savefig(path_energy, bbox_inches='tight')
                    plt.close(fig)
                    plots_to_upload['global/energy_heatmap'] = wandb.Image(path_energy)
                    
                    # 3B. Energy Histogram by Width
                    try:
                        by_width_energy = {}
                        for rec in global_measurements:
                            w = rec.get('width')
                            e = rec.get('energy_per_sample_mj')
                            if w is not None and e is not None:
                                by_width_energy.setdefault(w, []).append(e)
                                
                        if by_width_energy:
                            widths = sorted(by_width_energy.keys(), key=lambda x: (str(x)))
                            hist_data = [by_width_energy[w] for w in widths]
                            labels = [f"Width {w}" for w in widths]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(hist_data, bins=15, label=labels, alpha=0.7)
                            ax.set_xlabel('Energy per Sample (mJ)')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Energy Distribution by Network Width')
                            ax.legend()
                            path_eng_hist = os.path.join(tmpdir, 'energy_hist_by_width.png')
                            fig.tight_layout()
                            fig.savefig(path_eng_hist, bbox_inches='tight')
                            plt.close(fig)
                            plots_to_upload['global/energy_by_width_histogram'] = wandb.Image(path_eng_hist)
                    except Exception:
                        pass

                # 4. Power Heatmap
                if 'avg_power_mw' in df_global.columns and not df_global['avg_power_mw'].isna().all():
                    fig, ax = plt.subplots(figsize=(10, 8))
                    pivot_power = df_global.pivot_table(index='width', columns='batch_size', values='avg_power_mw', aggfunc='mean')
                    sns.heatmap(pivot_power, annot=True, fmt=".0f", cmap="Reds", ax=ax)
                    ax.set_title("Average Inference Power (mW) by Model Width and Batch Size")
                    ax.set_xlabel("Batch Size")
                    ax.set_ylabel("Network Width")
                    path_power = os.path.join(tmpdir, 'power_heatmap.png')
                    fig.savefig(path_power, bbox_inches='tight')
                    plt.close(fig)
                    plots_to_upload['global/power_heatmap'] = wandb.Image(path_power)

                # 5. Throughput per Watt (computed via throughput / (power/1000))
                if 'avg_power_mw' in df_global.columns and not df_global['avg_power_mw'].isna().all():
                    df_global['throughput_per_watt'] = df_global['throughput_samples_per_sec'] / (df_global['avg_power_mw'] / 1000.0)
                    # replace inf with NaN
                    df_global['throughput_per_watt'] = df_global['throughput_per_watt'].replace([np.inf, -np.inf], np.nan)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    pivot_tpw = df_global.pivot_table(index='width', columns='batch_size', values='throughput_per_watt', aggfunc='mean')
                    sns.heatmap(pivot_tpw, annot=True, fmt=".0f", cmap="Greens", ax=ax)
                    ax.set_title("Throughput per Watt (samples/sec/W)")
                    ax.set_xlabel("Batch Size")
                    ax.set_ylabel("Network Width")
                    path_tpw = os.path.join(tmpdir, 'throughput_per_watt_heatmap.png')
                    fig.savefig(path_tpw, bbox_inches='tight')
                    plt.close(fig)
                    plots_to_upload['global/throughput_per_watt_heatmap'] = wandb.Image(path_tpw)

                if wandb_online and plots_to_upload:
                    try:
                        retry_call(wandb.log, retries=3, backoff_sec=3, data=plots_to_upload)
                    except Exception as e:
                        print('Warning: failed to upload global plots to wandb:', e)
            finally:
                try:
                    shutil.rmtree(tmpdir)
                except Exception:
                    pass
    except Exception as e:
        print('Warning: failed to create global plots:', e)

    wandb.finish()


if __name__ == '__main__':
    main()
