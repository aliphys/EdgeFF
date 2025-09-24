#!/usr/bin/env python3
"""
Test script to verify CUDA availability and basic GPU operations
"""

import torch
import time
import numpy as np

def test_cuda_availability():
    """Test if CUDA is available and display GPU information"""
    print("=" * 50)
    print("CUDA AVAILABILITY TEST")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
        
        return True
    else:
        print("CUDA is not available. Will use CPU.")
        return False

def test_basic_gpu_operations():
    """Test basic tensor operations on GPU"""
    print("\n" + "=" * 50)
    print("BASIC GPU OPERATIONS TEST")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("Skipping GPU tests - CUDA not available")
        return False
    
    try:
        # Test tensor creation on GPU
        device = torch.device('cuda')
        print(f"Using device: {device}")
        
        # Create tensors
        x_cpu = torch.randn(1000, 1000)
        print(f"Created CPU tensor: {x_cpu.device}, shape: {x_cpu.shape}")
        
        # Move to GPU
        x_gpu = x_cpu.to(device)
        print(f"Moved to GPU: {x_gpu.device}, shape: {x_gpu.shape}")
        
        # Create tensor directly on GPU
        y_gpu = torch.randn(1000, 1000, device=device)
        print(f"Created GPU tensor: {y_gpu.device}, shape: {y_gpu.shape}")
        
        # Test operations
        z_gpu = torch.mm(x_gpu, y_gpu)
        print(f"Matrix multiplication result: {z_gpu.device}, shape: {z_gpu.shape}")
        
        # Test moving back to CPU
        z_cpu = z_gpu.cpu()
        print(f"Moved back to CPU: {z_cpu.device}, shape: {z_cpu.shape}")
        
        print("✓ All basic GPU operations successful!")
        return True
        
    except Exception as e:
        print(f"✗ GPU operations failed: {e}")
        return False

def test_performance_comparison():
    """Compare CPU vs GPU performance for matrix operations"""
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON TEST")
    print("=" * 50)
    
    size = 2000
    iterations = 5
    
    # CPU test
    print("Testing CPU performance...")
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    
    start_time = time.time()
    for _ in range(iterations):
        z_cpu = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU time ({iterations} iterations): {cpu_time:.3f} seconds")
    
    if torch.cuda.is_available():
        # GPU test
        print("Testing GPU performance...")
        device = torch.device('cuda')
        x_gpu = x_cpu.to(device)
        y_gpu = y_cpu.to(device)
        
        # Warm up GPU
        for _ in range(2):
            _ = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(iterations):
            z_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"GPU time ({iterations} iterations): {gpu_time:.3f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        
        return cpu_time, gpu_time
    else:
        print("GPU test skipped - CUDA not available")
        return cpu_time, None

if __name__ == "__main__":
    print("Testing CUDA setup for EdgeFF project")
    print("This will help verify GPU acceleration is working\n")
    
    # Run all tests
    cuda_available = test_cuda_availability()
    gpu_operations_ok = test_basic_gpu_operations()
    cpu_time, gpu_time = test_performance_comparison()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if cuda_available and gpu_operations_ok:
        print("✓ CUDA is properly configured and working!")
        if gpu_time:
            print(f"✓ GPU shows {cpu_time/gpu_time:.1f}x speedup over CPU")
        print("✓ Ready to implement GPU acceleration in training")
    else:
        print("✗ CUDA setup issues detected")
        print("  Please check CUDA installation and PyTorch GPU support")
    
    print("\nNext steps:")
    print("1. If CUDA works, proceed with GPU-enabled training")
    print("2. If CUDA fails, check drivers and PyTorch installation")
