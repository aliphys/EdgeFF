#!/usr/bin/env python3
"""Multi-purpose evaluation wrapper for basicRun."""

# This file is the CLI wrapper/entrypoint for evaluation workflows.
# It is intentionally separate from the package helper module
# `basicRun/evaluation.py`, which provides the underlying evaluation functions.

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import wandb

ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from basicRun import load_config, load_env, eval_with_inference_measurement
from basicRun.data import get_test_loader

COMMANDS = ['inference', 'energy', 'variance']


def load_model_from_artifact(model_path, device, onehot_max_value, is_color):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.device = device
    for layer in model.layers:
        layer.to(device)
    for softmax_layer in model.softmax_layers:
        softmax_layer.to(device)
    model.onehot_max_value = onehot_max_value
    model.is_color = is_color
    return model


def prepare_test_data(dataset_name, batch_size):
    test_loader, is_color = get_test_loader(dataset_name, batch_size=batch_size)
    inputs = torch.cat([d for d, _ in test_loader], dim=0)
    targets = torch.cat([t for _, t in test_loader], dim=0)
    return inputs, targets, is_color


def evaluate_inference(model, test_inputs, test_targets, batch_sizes, hw_monitor=None):
    device = model.device
    if hw_monitor:
        metrics = eval_with_inference_measurement(
            model,
            test_inputs,
            test_targets,
            hw_monitor=hw_monitor,
            set_name='test',
            batch_size=batch_sizes[0] if batch_sizes else None,
        )
        return metrics

    predictions = model.predict_one_pass(test_inputs, batch_size=test_inputs.shape[0])
    accuracy = (predictions.cpu() == test_targets.cpu()).float().mean().item()
    return {'test/accuracy': accuracy}


def evaluate_energy(model, test_inputs, test_targets, batch_sizes, hw_monitor=None):
    results = {}
    if hw_monitor:
        for batch_size in batch_sizes:
            metrics = eval_with_inference_measurement(
                model,
                test_inputs,
                test_targets,
                hw_monitor=hw_monitor,
                set_name='test',
                batch_size=batch_size,
            )
            results[f'test/accuracy_batch_{batch_size}'] = metrics.get('test/accuracy', 0)
            results[f'test/latency_per_sample_ms_batch_{batch_size}'] = metrics.get('test/inference_latency_per_sample_ms', 0)
            results[f'test/energy_per_sample_mj_batch_{batch_size}'] = metrics.get('test/inference_energy_per_sample_mj', 0)
            results[f'test/avg_power_mw_batch_{batch_size}'] = metrics.get('test/inference_avg_power_mw', 0)
            results[f'test/memory_mb_batch_{batch_size}'] = metrics.get('test/inference_memory_mb', 0)
    else:
        accuracy = evaluate_inference(model, test_inputs, test_targets, batch_sizes)['test/accuracy']
        results['test/accuracy'] = accuracy
    return results


def evaluate_variance(model, test_inputs, test_targets, batch_sizes, iterations, device):
    results = {}
    for batch_size in batch_sizes:
        latencies = []
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model.predict_one_pass(test_inputs, batch_size=batch_size)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start_time) * 1000)
        predictions = model.predict_one_pass(test_inputs, batch_size=batch_size)
        accuracy = (predictions.cpu() == test_targets.cpu()).float().mean().item()
        results[f'test/mean_latency_ms_batch_{batch_size}'] = float(sum(latencies) / len(latencies))
        results[f'test/std_latency_ms_batch_{batch_size}'] = float(torch.tensor(latencies).float().std().item())
        results[f'test/accuracy_batch_{batch_size}'] = accuracy
    return results


def load_dotenv_if_present():
    from dotenv import load_dotenv
    root_dir = PACKAGE_ROOT.parent
    dotenv_path = root_dir / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)


def main():
    parser = argparse.ArgumentParser(description='Run evaluation workflows for basicRun.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    for command in COMMANDS:
        sub = subparsers.add_parser(command, help=f'Run the {command} evaluation flow')
        sub.add_argument('--config', type=str, default='configs/eval.yaml', help='Path to YAML config file')
        if command == 'variance':
            sub.add_argument('--iterations', type=int, default=5, help='Number of repeated inferences per batch size')

    args = parser.parse_args()
    config = load_config(args.config)
    load_env()
    load_dotenv_if_present()

    wandb.init(project=config.get('project', 'edgeff-refactor'), job_type='eval')
    api = wandb.Api()
    project_name = config.get('project', 'edgeff-refactor')
    sweep_id = config.get('sweep_id')
    if not sweep_id:
        raise ValueError('sweep_id must be specified in config')

    dataset_name = config.get('dataset', 'MNIST')
    batch_sizes = config.get('inference_batch_sizes', [1, 2, 8, 16, 32, 64, 128, 256, 512])
    test_inputs, test_targets, is_color = prepare_test_data(dataset_name, batch_size=max(batch_sizes))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    try:
        from basicRun import TegratsMonitor
    except Exception:
        TegratsMonitor = None

    hw_monitor = None
    if args.command == 'energy' and TegratsMonitor is not None:
        try:
            hw_monitor = TegratsMonitor(interval_ms=config.get('hw_interval_ms', 500))
            power_mode = hw_monitor.get_power_mode()
            if power_mode:
                wandb.run.config.update({'system/power_mode': power_mode})
            hw_monitor.start()
        except Exception as exc:
            print(f'Hardware monitoring unavailable: {exc}')
            hw_monitor = None

    sweep = api.sweep(f'{project_name}/{sweep_id}')
    for run in sweep.runs:
        run_id = run.id
        print(f'Evaluating model from run {run_id}')
        model_artifact = next((art for art in run.logged_artifacts() if art.type == 'model'), None)
        if model_artifact is None:
            print(f'No model artifact for run {run_id}, skipping')
            continue

        artifact_dir = model_artifact.download()
        model_path = os.path.join(artifact_dir, 'temp_')
        model = load_model_from_artifact(model_path, device, onehot_max_value=10.0, is_color=is_color)

        width = model_artifact.metadata.get('width', 0) if model_artifact.metadata else 0

        if args.command == 'inference':
            metrics = evaluate_inference(model, test_inputs, test_targets, batch_sizes, hw_monitor=None)
        elif args.command == 'energy':
            metrics = evaluate_energy(model, test_inputs, test_targets, batch_sizes, hw_monitor=hw_monitor)
        else:
            metrics = evaluate_variance(model, test_inputs, test_targets, batch_sizes, args.iterations, device)

        metrics['run_id'] = run_id
        metrics['width'] = width
        metrics['sweep_id'] = sweep_id
        wandb.log(metrics)

    if hw_monitor:
        hw_monitor.stop()
    wandb.finish()


if __name__ == '__main__':
    main()
