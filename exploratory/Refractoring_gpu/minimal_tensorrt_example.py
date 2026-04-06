#!/usr/bin/env python3
"""
Minimal TensorRT Inference Example
===================================

This is the simplest possible example of using TensorRT with the Forward-Forward model.
Great for understanding the basics or quick testing.

Usage:
    python minimal_tensorrt_example.py
"""

import torch
import torch_tensorrt
from Train import Net


def main():
    print("Minimal TensorRT Inference Example")
    print("=" * 60)
    
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for TensorRT")
        return
    
    # 2. Create model (or load from checkpoint)
    print("\n1. Creating model...")
    layers = [784, 200, 200, 10]
    model = Net(dims=layers, device=device)
    model.eval()
    print(f"   Model architecture: {layers}")
    
    # 3. Create example input
    print("\n2. Preparing example input...")
    batch_size = 32
    example_input = torch.randn(batch_size, 784).to(device)
    print(f"   Input shape: {example_input.shape}")
    
    # 4. Wrap model for TensorRT
    print("\n3. Wrapping model...")
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, model, batch_size):
            super().__init__()
            self.model = model
            self.batch_size = batch_size
            
        def forward(self, x):
            return self.model.predict_one_pass(x, self.batch_size)
    
    wrapped_model = InferenceWrapper(model, batch_size)
    
    # 5. Test PyTorch inference first
    print("\n4. Testing PyTorch inference...")
    with torch.no_grad():
        pytorch_output = wrapped_model(example_input)
    print(f"   PyTorch output shape: {pytorch_output.shape}")
    print(f"   PyTorch predictions: {pytorch_output[:5]}")  # Show first 5
    
    # 6. Compile with TensorRT
    print("\n5. Compiling with TensorRT...")
    print("   (This may take a minute...)")
    try:
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
            enabled_precisions={torch.float32},
            workspace_size=1 << 30,  # 1GB
            truncate_long_and_double=True,
        )
        print("   ✓ TensorRT compilation successful!")
    except Exception as e:
        print(f"   ✗ TensorRT compilation failed: {e}")
        return
    
    # 7. Test TensorRT inference
    print("\n6. Testing TensorRT inference...")
    with torch.no_grad():
        trt_output = trt_model(example_input)
    print(f"   TensorRT output shape: {trt_output.shape}")
    print(f"   TensorRT predictions: {trt_output[:5]}")  # Show first 5
    
    # 8. Compare outputs
    print("\n7. Comparing outputs...")
    max_diff = torch.max(torch.abs(pytorch_output - trt_output)).item()
    print(f"   Max difference: {max_diff}")
    if max_diff < 0.01:
        print("   ✓ Outputs match closely!")
    else:
        print(f"   ⚠ Outputs differ by {max_diff}")
    
    # 9. Quick benchmark
    print("\n8. Quick performance comparison...")
    import time
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = wrapped_model(example_input)
            _ = trt_model(example_input)
    
    # Time PyTorch
    torch.cuda.synchronize()
    num_runs = 100
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = wrapped_model(example_input)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Time TensorRT
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = trt_model(example_input)
    torch.cuda.synchronize()
    trt_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"   PyTorch:   {pytorch_time:.3f} ms/batch")
    print(f"   TensorRT:  {trt_time:.3f} ms/batch")
    print(f"   Speedup:   {pytorch_time/trt_time:.2f}x")
    
    print("\n" + "=" * 60)
    print("✓ Success! TensorRT inference is working.")
    print("=" * 60)
    
    # 10. Optional: Save the TensorRT model
    save_model = input("\nSave TensorRT model? (y/n): ").strip().lower()
    if save_model == 'y':
        output_path = "model_trt.ts"
        torch.jit.save(trt_model, output_path)
        print(f"✓ Saved TensorRT model to {output_path}")
        print(f"  Load with: trt_model = torch.jit.load('{output_path}')")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
