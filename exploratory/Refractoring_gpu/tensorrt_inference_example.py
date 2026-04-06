"""
TensorRT Inference Example
==========================

This example demonstrates how to perform inference using Torch-TensorRT
on the Forward-Forward network model trained in this project.

Requirements:
    - torch_tensorrt (install: pip install torch-tensorrt)
    - A trained model checkpoint

Usage:
    python tensorrt_inference_example.py --model-path path/to/model.pth --layers "784,200,200,10"
"""

import torch
import torch_tensorrt
import argparse
from pathlib import Path
import time
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

# Import the model architecture from Train.py
from Train import Net


def load_model(model_path, layers, device='cuda'):
    """
    Load a trained Forward-Forward model from checkpoint.
    
    Args:
        model_path: Path to the .pth model checkpoint
        layers: List of layer dimensions (e.g., [784, 200, 200, 10])
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    model = Net(dims=layers, device=device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    return model


def compile_with_tensorrt(model, example_input, device='cuda'):
    """
    Compile the model using Torch-TensorRT for optimized inference.
    
    Args:
        model: PyTorch model
        example_input: Example input tensor for tracing
        device: Device to compile for
        
    Returns:
        TensorRT compiled model
    """
    print("\nCompiling model with Torch-TensorRT...")
    print(f"Input shape: {example_input.shape}")
    
    # Move example input to device
    example_input = example_input.to(device)
    
    # Create a wrapper that only exposes predict_one_pass method
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, model, batch_size):
            super().__init__()
            self.model = model
            self.batch_size = batch_size
            
        def forward(self, x):
            return self.model.predict_one_pass(x, self.batch_size)
    
    # Wrap the model
    batch_size = example_input.shape[0]
    wrapped_model = InferenceWrapper(model, batch_size)
    wrapped_model.eval()
    
    try:
        # Compile with Torch-TensorRT
        trt_model = torch_tensorrt.compile(
            wrapped_model,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=example_input.shape,
                    opt_shape=example_input.shape,
                    max_shape=example_input.shape,
                    dtype=torch.float32
                )
            ],
            enabled_precisions={torch.float32},  # Can also try torch.float16 for faster inference
            workspace_size=1 << 30,  # 1GB workspace
            truncate_long_and_double=True,
        )
        print("✓ TensorRT compilation successful!")
        return trt_model
        
    except Exception as e:
        print(f"✗ TensorRT compilation failed: {e}")
        print("Falling back to regular PyTorch inference")
        return wrapped_model


def benchmark_inference(model, test_loader, device='cuda', num_batches=10):
    """
    Benchmark inference performance.
    
    Args:
        model: Model to benchmark (TensorRT or PyTorch)
        test_loader: DataLoader for test data
        device: Device to run inference on
        num_batches: Number of batches to benchmark
        
    Returns:
        Dictionary with performance metrics
    """
    print(f"\nBenchmarking inference ({num_batches} batches)...")
    
    latencies = []
    correct = 0
    total = 0
    
    # Warmup
    for i, (data, target) in enumerate(test_loader):
        if i >= 3:
            break
        data = data.to(device)
        with torch.no_grad():
            _ = model(data)
    
    # Benchmark
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_batches:
                break
                
            data = data.to(device)
            target = target.to(device)
            
            # Measure inference time
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            output = model(data)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Calculate accuracy
            correct += (output == target).sum().item()
            total += target.size(0)
    
    metrics = {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'throughput_samples_per_sec': total / (sum(latencies) / 1000),
        'accuracy': correct / total if total > 0 else 0,
        'total_samples': total
    }
    
    return metrics


def print_metrics(metrics, model_type):
    """Print benchmark metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{model_type} Performance Metrics")
    print(f"{'='*60}")
    print(f"Mean Latency:    {metrics['mean_latency_ms']:.3f} ± {metrics['std_latency_ms']:.3f} ms")
    print(f"Min Latency:     {metrics['min_latency_ms']:.3f} ms")
    print(f"Max Latency:     {metrics['max_latency_ms']:.3f} ms")
    print(f"Throughput:      {metrics['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"Accuracy:        {metrics['accuracy']*100:.2f}% ({metrics['total_samples']} samples)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='TensorRT inference example for Forward-Forward networks')
    parser.add_argument('--model-path', type=str, required=False,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--layers', type=str, default='784,200,200,10',
                        help='Comma-separated layer dimensions')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num-batches', type=int, default=20,
                        help='Number of batches to benchmark')
    parser.add_argument('--no-tensorrt', action='store_true',
                        help='Skip TensorRT compilation (use PyTorch only)')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST'],
                        help='Dataset to use for testing')
    args = parser.parse_args()
    
    # Parse layer dimensions
    layers = [int(x.strip()) for x in args.layers.split(',')]
    print(f"Model architecture: {layers}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. TensorRT requires CUDA.")
        return
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load test dataset
    print("\nLoading MNIST test dataset...")
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])
    test_dataset = MNIST('./data/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"✓ Loaded {len(test_dataset)} test samples")
    
    # Load model or create a new one for demonstration
    if args.model_path:
        model = load_model(args.model_path, layers, device=device)
    else:
        print("\nNo model path provided. Creating a new (untrained) model for demonstration...")
        print("Note: For meaningful results, provide a trained model with --model-path")
        model = Net(dims=layers, device=device)
        model.eval()
    
    # Get example input for TensorRT compilation
    example_batch, _ = next(iter(test_loader))
    example_input = example_batch.to(device)
    
    # Benchmark PyTorch inference
    print("\n" + "="*60)
    print("PYTORCH INFERENCE")
    print("="*60)
    
    class PyTorchWrapper(torch.nn.Module):
        def __init__(self, model, batch_size):
            super().__init__()
            self.model = model
            self.batch_size = batch_size
            
        def forward(self, x):
            return self.model.predict_one_pass(x, self.batch_size)
    
    pytorch_model = PyTorchWrapper(model, args.batch_size)
    pytorch_metrics = benchmark_inference(pytorch_model, test_loader, device, args.num_batches)
    print_metrics(pytorch_metrics, "PyTorch")
    
    # Compile and benchmark TensorRT inference
    if not args.no_tensorrt:
        print("\n" + "="*60)
        print("TENSORRT INFERENCE")
        print("="*60)
        
        trt_model = compile_with_tensorrt(model, example_input, device=device)
        trt_metrics = benchmark_inference(trt_model, test_loader, device, args.num_batches)
        print_metrics(trt_metrics, "TensorRT")
        
        # Print speedup
        speedup = pytorch_metrics['mean_latency_ms'] / trt_metrics['mean_latency_ms']
        throughput_increase = (trt_metrics['throughput_samples_per_sec'] / 
                              pytorch_metrics['throughput_samples_per_sec'] - 1) * 100
        
        print(f"{'='*60}")
        print(f"PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"Speedup:              {speedup:.2f}x faster")
        print(f"Throughput increase:  {throughput_increase:+.1f}%")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
