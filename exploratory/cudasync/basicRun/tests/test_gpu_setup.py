#!/usr/bin/env python3
"""
Quick test to verify GPU setup is working correctly.
This script tests device detection and basic tensor operations on GPU.
"""

import torch
import sys

def test_gpu_setup():
    print("=" * 60)
    print("GPU Setup Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
        
        props = torch.cuda.get_device_properties(0)
        print(f"   Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
    else:
        print("   ⚠️  CUDA is not available. Running on CPU.")
    
    # Test device creation
    print(f"\n2. Device Configuration:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Selected Device: {device}")
    
    # Test tensor creation on device
    print(f"\n3. Tensor Operations Test:")
    try:
        # Create tensor on device
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        
        # Perform operation
        z = torch.mm(x, y)
        
        print(f"   ✓ Created tensors on {device}")
        print(f"   ✓ Matrix multiplication successful")
        print(f"   ✓ Result device: {z.device}")
        
        # Test moving to CPU
        z_cpu = z.cpu()
        print(f"   ✓ Transfer to CPU successful")
        
    except Exception as e:
        print(f"   ✗ Error during tensor operations: {e}")
        return False
    
    # Test simple neural network layer
    print(f"\n4. Neural Network Layer Test:")
    try:
        layer = torch.nn.Linear(100, 50).to(device)
        x = torch.randn(32, 100).to(device)
        output = layer(x)
        
        print(f"   ✓ Created Linear layer on {device}")
        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output device: {output.device}")
        
    except Exception as e:
        print(f"   ✗ Error during NN layer test: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_gpu_setup()
    sys.exit(0 if success else 1)
