#!/usr/bin/env python3
"""
Quick test to verify the TensorRT example can be imported and basic functionality works.
"""

import sys
import torch

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torch_tensorrt
        print(f"✓ Torch-TensorRT version: {torch_tensorrt.__version__}")
    except ImportError as e:
        print(f"✗ Torch-TensorRT not available: {e}")
        print("  Install with: pip install torch-tensorrt")
        return False
    
    try:
        from Train import Net
        print("✓ Train.Net imported successfully")
    except ImportError as e:
        print(f"✗ Train import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    try:
        from Train import Net
        layers = [784, 200, 200, 10]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(dims=layers, device=device)
        model.eval()
        print(f"✓ Model created successfully on {device}")
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 784).to(device)
        with torch.no_grad():
            output = model.predict_one_pass(x, batch_size=batch_size)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("✗ CUDA not available - TensorRT requires CUDA")
        return False


def main():
    print("="*60)
    print("TensorRT Inference Example - Environment Check")
    print("="*60)
    
    results = {
        'Imports': test_imports(),
        'CUDA': test_cuda(),
        'Model Creation': test_model_creation(),
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("="*60)
    
    if all_passed:
        print("\n✓ All checks passed! You can run the TensorRT example.")
        print("\nTry: python tensorrt_inference_example.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
