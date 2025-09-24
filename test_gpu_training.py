#!/usr/bin/env python3
"""
Minimal test to verify GPU-accelerated training works with small dataset
"""

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
import os

# Add current directory to path to import Train module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Train import Net, overlay_y_on_x, overlay_on_x_neutral, get_device, prepare_data_for_training

def create_test_data(num_samples=1000):
    """Create small test dataset for validation"""
    print(f"Creating test dataset with {num_samples} samples...")
    
    # Load MNIST data
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])
    
    dataset = MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    
    # Get a small batch
    inputs, targets = next(iter(loader))
    
    print(f"✓ Data created: inputs shape {inputs.shape}, targets shape {targets.shape}")
    return inputs, targets

def test_model_creation_and_gpu_movement():
    """Test that model can be created and moved to GPU"""
    print("\n" + "=" * 50)
    print("MODEL CREATION AND GPU MOVEMENT TEST")
    print("=" * 50)
    
    try:
        device = get_device()
        dims = [784, 100, 100]  # Simplified architecture
        
        # Create model
        model = Net(dims, goodness_threshold=2.0, confidence_threshold_multiplier=1.0, device=device)
        model = model.to(device)
        
        print(f"✓ Model created and moved to {device}")
        
        # Test a forward pass with dummy data
        dummy_input = torch.randn(10, 784, device=device)
        dummy_neutral = overlay_on_x_neutral(dummy_input)
        
        # Test prediction
        prediction = model.predict_one_pass(dummy_neutral, batch_size=10)
        print(f"✓ Forward pass successful, prediction shape: {prediction.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_minimal_training():
    """Test minimal training loop with small dataset"""
    print("\n" + "=" * 50)
    print("MINIMAL TRAINING TEST")
    print("=" * 50)
    
    try:
        # Create small test dataset
        inputs, targets = create_test_data(num_samples=500)
        
        # Prepare training data
        x_pos = overlay_y_on_x(inputs, targets)
        
        # Create negative samples
        y_neg = targets.clone()
        for idx, y_samp in enumerate(targets):
            allowed_indices = list(range(10))
            allowed_indices.remove(y_samp.item())
            y_neg[idx] = torch.tensor(np.random.choice(allowed_indices))
        x_neg = overlay_y_on_x(inputs, y_neg)
        
        # Neutral samples
        x_neutral = overlay_on_x_neutral(inputs)
        
        # Move to device
        device = get_device()
        x_pos, x_neg, x_neutral, targets = prepare_data_for_training(x_pos, x_neg, x_neutral, targets, device)
        
        # Create model
        dims = [784, 100, 100]  # Simplified architecture
        model = Net(dims, goodness_threshold=2.0, confidence_threshold_multiplier=1.0, device=device)
        model = model.to(device)
        
        print("Starting minimal training (3 epochs)...")
        
        # Train layers for a few epochs
        num_epochs = 3
        for epoch in range(num_epochs):
            print(f"  Epoch {epoch + 1}/{num_epochs}")
            model.train(x_pos, x_neg)
        
        print("✓ Layer training completed")
        
        # Train softmax layers
        batch_size = len(inputs)
        for epoch in range(num_epochs):
            print(f"  Softmax epoch {epoch + 1}/{num_epochs}")
            model.train_softmax_layer(x_neutral, targets, batch_size, dims)
        
        print("✓ Softmax training completed")
        
        # Test prediction
        with torch.no_grad():
            predictions = model.predict_one_pass(x_neutral[:10], batch_size=10)
            accuracy = (predictions.cpu() == targets[:10].cpu()).float().mean()
            print(f"✓ Training successful! Sample accuracy: {accuracy:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        return False

def test_performance_comparison():
    """Compare CPU vs GPU training performance"""
    print("\n" + "=" * 50)
    print("TRAINING PERFORMANCE COMPARISON")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("GPU not available, skipping performance comparison")
        return
    
    try:
        # Create test data
        inputs, targets = create_test_data(num_samples=1000)
        
        # Prepare training data
        x_pos = overlay_y_on_x(inputs, targets)
        y_neg = targets.clone()
        for idx, y_samp in enumerate(targets):
            allowed_indices = list(range(10))
            allowed_indices.remove(y_samp.item())
            y_neg[idx] = torch.tensor(np.random.choice(allowed_indices))
        x_neg = overlay_y_on_x(inputs, y_neg)
        
        dims = [784, 100]  # Simple for speed
        
        # CPU Training
        print("Testing CPU training...")
        model_cpu = Net(dims, device=torch.device('cpu'))
        x_pos_cpu, x_neg_cpu = x_pos.cpu(), x_neg.cpu()
        
        start_time = time.time()
        for epoch in range(2):
            model_cpu.train(x_pos_cpu, x_neg_cpu)
        cpu_time = time.time() - start_time
        print(f"CPU training time: {cpu_time:.3f} seconds")
        
        # GPU Training
        print("Testing GPU training...")
        device = torch.device('cuda')
        model_gpu = Net(dims, device=device).to(device)
        x_pos_gpu, x_neg_gpu = x_pos.to(device), x_neg.to(device)
        
        # Warm up
        model_gpu.train(x_pos_gpu[:100], x_neg_gpu[:100])
        torch.cuda.synchronize()
        
        start_time = time.time()
        for epoch in range(2):
            model_gpu.train(x_pos_gpu, x_neg_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU training time: {gpu_time:.3f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        
        return cpu_time, gpu_time
        
    except Exception as e:
        print(f"✗ Performance comparison failed: {e}")
        return None, None

if __name__ == "__main__":
    print("Testing GPU-accelerated training for EdgeFF")
    print("This test uses a minimal dataset and simplified architecture\n")
    
    # Run all tests
    model_test_passed = test_model_creation_and_gpu_movement()
    training_test_passed = test_minimal_training()
    cpu_time, gpu_time = test_performance_comparison()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if model_test_passed and training_test_passed:
        print("✓ All tests passed! GPU training is working correctly")
        if gpu_time and cpu_time:
            print(f"✓ GPU shows {cpu_time/gpu_time:.1f}x speedup over CPU")
        print("✓ Ready to run full training with GPU acceleration")
    else:
        print("✗ Some tests failed")
        if not model_test_passed:
            print("  - Model creation/GPU movement failed")
        if not training_test_passed:
            print("  - Training test failed")
    
    print("\nNext steps:")
    print("1. If tests pass, run the full training with GPU")
    print("2. If tests fail, check the error messages above")
    print("3. Consider using larger batch sizes to maximize GPU utilization")
