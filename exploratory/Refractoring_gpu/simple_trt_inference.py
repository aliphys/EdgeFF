import os
import time
import argparse
import torch
import wandb
import types

try:
    import torch_tensorrt as trt
except ImportError:
    print("Warning: torch_tensorrt not found. Please install it.")
    trt = None

# We can reuse the TRTWrapper from measure_trt_latency to handle the custom model logic
from measure_trt_latency import TRTWrapper

def main():
    parser = argparse.ArgumentParser(description="Simple TRT inference measurement.")
    parser.add_argument('--run_path', type=str, help='W&B run path, e.g. user/project/run_id')
    parser.add_argument('--sweep_path', type=str, help='W&B sweep path, e.g. user/project/sweep_id (will pick the first run)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--warmup', type=int, default=50, help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=200, help='Number of measurement iterations')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    args = parser.parse_args()

    if not args.run_path and not args.sweep_path:
        print("Please provide either --run_path or --sweep_path.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA is required for TensorRT inference.")
        return

    api = wandb.Api()
    
    # 1. Download model from W&B
    if args.sweep_path:
        print(f"Fetching sweep {args.sweep_path} from W&B...")
        sweep = api.sweep(args.sweep_path)
        runs = sweep.runs
        if not runs:
            print("No runs found in this sweep.")
            return
        run = runs[0]
        print(f"Selected first run from sweep: {run.path}")
    else:
        print(f"Fetching run {args.run_path} from W&B...")
        run = api.run(args.run_path)
    
    model_artifact = None
    for art in run.logged_artifacts():
        if art.type == 'model':
            model_artifact = art
            break
            
    if model_artifact is None:
        print("No model artifact found in this run.")
        return
        
    print("Downloading model artifact...")
    artifact_dir = model_artifact.download()
    model_path = os.path.join(artifact_dir, 'temp_')
    
    # 2. Load model
    print("Loading PyTorch model...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.device = device
    for layer in model.layers:
        layer.to(device)
    for s in model.softmax_layers:
        s.to(device)
        
    # Monkey-patch modules' train methods to avoid errors during TRT compilation
    def _make_train_override(m):
        def _train(self, mode=True):
            return torch.nn.Module.train(self, mode)
        return types.MethodType(_train, m)

    for m in model.modules():
        try:
            m.train = _make_train_override(m)
        except Exception:
            pass

    # 3. Compiling with Torch-TensorRT
    print("Compiling model with Torch-TensorRT...")
    input_dim = model.layers[0].in_features
    input_dtype = torch.float16 if args.fp16 else torch.float32
    
    wrapper = TRTWrapper(model)
    wrapper.eval()
    
    trt_input = trt.Input(
        shape=[args.batch_size, input_dim],
        dtype=input_dtype,
    )
    
    enabled_precisions = {torch.float32}
    if args.fp16:
        enabled_precisions.add(torch.float16)
        
    trt_mod = trt.compile(
        wrapper,
        inputs=[trt_input],
        enabled_precisions=enabled_precisions,
        workspace_size=1 << 28,
    )
    print("Model compiled successfully.")

    # 4. Measure Inference Time
    print(f"Running inference (Batch Size: {args.batch_size}, Warmup: {args.warmup}, Iters: {args.iters})...")
    dummy_input = torch.randn((args.batch_size, input_dim), device=device, dtype=input_dtype)
    
    # Warmup
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        _ = trt_mod(dummy_input)
    torch.cuda.synchronize()

    # Measurement
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]

    for i in range(args.iters):
        start_events[i].record()
        _ = trt_mod(dummy_input)
        end_events[i].record()
        
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_latency = sum(times) / args.iters
    throughput = (args.batch_size / (avg_latency / 1000.0)) if avg_latency > 0 else 0

    print("-" * 40)
    print("Results:")
    print(f"Batch Size  : {args.batch_size}")
    print(f"Precision   : {'FP16' if args.fp16 else 'FP32'}")
    print(f"Avg Latency : {avg_latency:.4f} ms")
    print(f"Throughput  : {throughput:.2f} samples/sec")
    print("-" * 40)

if __name__ == '__main__':
    main()
