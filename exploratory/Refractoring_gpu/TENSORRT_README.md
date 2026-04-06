# TensorRT Inference Example

This example demonstrates how to use Torch-TensorRT to optimize and accelerate inference for the Forward-Forward network model.

## Overview

The script:
1. Loads a trained Forward-Forward model (or creates an untrained one for demo)
2. Compiles the model with Torch-TensorRT for GPU optimization
3. Benchmarks both PyTorch and TensorRT inference
4. Compares performance metrics (latency, throughput, accuracy)

## Requirements

### Install Torch-TensorRT

For NVIDIA Jetson devices:
```bash
# Install from PyPI (if available for your platform)
pip install torch-tensorrt

# Or build from source if needed
# Follow: https://github.com/pytorch/TensorRT
```

For other platforms with CUDA:
```bash
pip install torch-tensorrt
```

## Usage

### Basic Usage (without trained model)

Run with a randomly initialized model to test the compilation pipeline:

```bash
python tensorrt_inference_example.py
```

This will:
- Create an untrained model with default architecture (784, 200, 200, 10)
- Download MNIST test data
- Benchmark PyTorch vs TensorRT inference

### With a Trained Model

If you have a trained model checkpoint:

```bash
python tensorrt_inference_example.py \
    --model-path path/to/your/model.pth \
    --layers "784,200,200,10"
```

### Advanced Options

```bash
python tensorrt_inference_example.py \
    --model-path checkpoints/best_model.pth \
    --layers "784,500,500,10" \
    --batch-size 64 \
    --num-batches 50 \
    --dataset MNIST
```

### Skip TensorRT (PyTorch only)

To only benchmark PyTorch inference without TensorRT compilation:

```bash
python tensorrt_inference_example.py --no-tensorrt
```

## Arguments

- `--model-path`: Path to trained model checkpoint (optional)
- `--layers`: Comma-separated layer dimensions (default: "784,200,200,10")
- `--batch-size`: Batch size for inference (default: 32)
- `--num-batches`: Number of batches to benchmark (default: 20)
- `--no-tensorrt`: Skip TensorRT compilation
- `--dataset`: Dataset to use (default: MNIST)

## Expected Output

```
Model architecture: [784, 200, 200, 10]
Using device: cuda
GPU: NVIDIA Jetson Orin

Loading MNIST test dataset...
✓ Loaded 10000 test samples

============================================================
PYTORCH INFERENCE
============================================================

Benchmarking inference (20 batches)...

============================================================
PyTorch Performance Metrics
============================================================
Mean Latency:    2.456 ± 0.123 ms
Min Latency:     2.301 ms
Max Latency:     2.789 ms
Throughput:      13027.4 samples/sec
Accuracy:        98.13% (640 samples)
============================================================

============================================================
TENSORRT INFERENCE
============================================================

Compiling model with Torch-TensorRT...
Input shape: torch.Size([32, 784])
✓ TensorRT compilation successful!

Benchmarking inference (20 batches)...

============================================================
TensorRT Performance Metrics
============================================================
Mean Latency:    1.234 ± 0.089 ms
Min Latency:     1.156 ms
Max Latency:     1.423 ms
Throughput:      25945.2 samples/sec
Accuracy:        98.13% (640 samples)
============================================================

============================================================
PERFORMANCE COMPARISON
============================================================
Speedup:              1.99x faster
Throughput increase:  +99.2%
============================================================
```

## How It Works

### Model Wrapping

The script wraps the Forward-Forward model's `predict_one_pass` method in a simple `forward()` method that TensorRT can compile:

```python
class InferenceWrapper(torch.nn.Module):
    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        
    def forward(self, x):
        return self.model.predict_one_pass(x, self.batch_size)
```

### TensorRT Compilation

The model is compiled with specific input shapes and precision:

```python
trt_model = torch_tensorrt.compile(
    wrapped_model,
    inputs=[torch_tensorrt.Input(
        min_shape=example_input.shape,
        opt_shape=example_input.shape,
        max_shape=example_input.shape,
        dtype=torch.float32
    )],
    enabled_precisions={torch.float32},  # Can try torch.float16 for faster inference
    workspace_size=1 << 30,  # 1GB workspace
)
```

### Benchmarking

Both models are benchmarked with:
- 3 warmup iterations
- Multiple batches for stable measurements
- CUDA synchronization for accurate timing
- Accuracy verification

## Optimization Tips

1. **Use FP16 precision** for faster inference (if accuracy permits):
   ```python
   enabled_precisions={torch.float16, torch.float32}
   ```

2. **Optimize batch size** for your hardware:
   - Larger batches = better throughput
   - Smaller batches = lower latency

3. **Fixed input shapes** work best with TensorRT:
   - Dynamic shapes have overhead
   - Use consistent batch sizes if possible

4. **Model architecture matters**:
   - Wider layers benefit more from TensorRT
   - More layers = more optimization opportunities

## Troubleshooting

### TensorRT compilation fails

If compilation fails, the script falls back to PyTorch inference. Common causes:
- Unsupported operations in the model
- Memory constraints
- TensorRT version incompatibility

### CUDA out of memory

Reduce `--batch-size` or ensure no other GPU processes are running:
```bash
nvidia-smi  # Check GPU memory usage
```

### Lower than expected speedup

- Ensure GPU is in performance mode (Jetson devices)
- Try FP16 precision
- Increase batch size
- Check for CPU-GPU data transfer bottlenecks

## References

- [Torch-TensorRT Documentation](https://pytorch.org/TensorRT/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Forward-Forward Algorithm Paper](https://arxiv.org/abs/2212.13345)
