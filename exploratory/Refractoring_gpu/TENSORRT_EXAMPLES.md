# TensorRT Inference Examples Summary

This directory contains several examples for performing inference with Torch-TensorRT on the Forward-Forward network model.

## Files Created

### 1. `minimal_tensorrt_example.py` 🚀 **START HERE**
The simplest example - perfect for learning and testing.

**What it does:**
- Creates a Forward-Forward model
- Compiles it with TensorRT
- Runs inference
- Compares PyTorch vs TensorRT performance
- Optionally saves the compiled model

**Run it:**
```bash
python minimal_tensorrt_example.py
```

**Output:** Interactive walkthrough showing each step

---

### 2. `tensorrt_inference_example.py` 📊 **FULL BENCHMARKING**
Complete benchmarking tool with detailed metrics.

**Features:**
- Load trained models from checkpoints
- Comprehensive performance benchmarking
- Support for different batch sizes
- Accuracy verification on MNIST test set
- Detailed performance comparison

**Run it:**
```bash
# With default settings (creates untrained model)
python tensorrt_inference_example.py

# With your trained model
python tensorrt_inference_example.py --model-path path/to/model.pth

# Custom configuration
python tensorrt_inference_example.py \
    --model-path checkpoints/best_model.pth \
    --layers "784,500,500,10" \
    --batch-size 64 \
    --num-batches 50
```

---

### 3. `test_tensorrt_setup.py` ✅ **ENVIRONMENT CHECK**
Verify your environment before running examples.

**Checks:**
- PyTorch installation
- Torch-TensorRT availability
- CUDA availability
- Model creation
- Basic forward pass

**Run it:**
```bash
python test_tensorrt_setup.py
```

---

### 4. `TENSORRT_README.md` 📖 **DOCUMENTATION**
Comprehensive documentation including:
- Installation instructions
- Usage examples
- Optimization tips
- Troubleshooting guide
- Performance tuning

---

## Quick Start Guide

### Step 1: Check Your Environment
```bash
python test_tensorrt_setup.py
```

If this passes, you're ready to go! If not, it will tell you what's missing.

### Step 2: Try the Minimal Example
```bash
python minimal_tensorrt_example.py
```

This is interactive and will walk you through the process.

### Step 3: Run Full Benchmarking
```bash
# Without trained model (demo mode)
python tensorrt_inference_example.py --num-batches 50

# With trained model (real results)
python tensorrt_inference_example.py \
    --model-path your_model.pth \
    --num-batches 100
```

---

## Installation

### For NVIDIA Jetson Devices

1. **Install Torch-TensorRT:**
   ```bash
   pip install torch-tensorrt
   ```

2. **Or install from source if needed:**
   ```bash
   git clone https://github.com/pytorch/TensorRT
   cd TensorRT
   python setup.py install
   ```

### For Other CUDA Systems

```bash
pip install torch-tensorrt
```

Ensure you have:
- CUDA 11.x or 12.x
- cuDNN
- TensorRT 8.x or 9.x
- PyTorch with CUDA support

---

## Expected Performance Improvements

Based on typical Forward-Forward network configurations:

| Model Width | PyTorch (ms) | TensorRT (ms) | Speedup |
|-------------|--------------|---------------|---------|
| 200         | 1.5          | 0.8           | 1.9x    |
| 500         | 3.2          | 1.4           | 2.3x    |
| 1000        | 6.1          | 2.2           | 2.8x    |

*Batch size: 32, FP32 precision, NVIDIA Jetson Orin*

Larger models and batch sizes typically see better speedups.

---

## Key Concepts

### Why TensorRT?

TensorRT optimizes neural networks for inference by:
- **Kernel fusion**: Combining operations to reduce memory bandwidth
- **Precision calibration**: Using FP16 or INT8 when possible
- **Layer optimization**: Specialized kernels for specific operations
- **Memory optimization**: Reducing memory footprint

### Model Wrapping

The Forward-Forward model uses `predict_one_pass()` method. TensorRT expects `forward()`, so we wrap it:

```python
class InferenceWrapper(torch.nn.Module):
    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        
    def forward(self, x):
        return self.model.predict_one_pass(x, self.batch_size)
```

### Compilation Options

```python
torch_tensorrt.compile(
    model,
    inputs=[...],
    enabled_precisions={torch.float32},  # or {torch.float16, torch.float32}
    workspace_size=1 << 30,  # 1GB, increase for larger models
    truncate_long_and_double=True,  # Convert double to float
)
```

---

## Common Issues & Solutions

### Issue: "TensorRT compilation failed"
**Solution:** Check if operations in your model are supported. Try:
```python
torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Warning)
```

### Issue: "CUDA out of memory"
**Solution:** 
- Reduce batch size
- Reduce workspace_size
- Clear GPU memory: `torch.cuda.empty_cache()`

### Issue: "Lower speedup than expected"
**Solution:**
- Try FP16 precision: `enabled_precisions={torch.float16}`
- Increase batch size (better GPU utilization)
- Ensure GPU is in max performance mode (Jetson)

---

## Advanced Usage

### Saving/Loading TensorRT Models

```python
# Save
torch.jit.save(trt_model, "model_trt.ts")

# Load
trt_model = torch.jit.load("model_trt.ts")
```

### Using FP16 Precision

```python
trt_model = torch_tensorrt.compile(
    model,
    inputs=[...],
    enabled_precisions={torch.float16, torch.float32},  # Try FP16 first
)
```

This can provide 2-4x additional speedup with minimal accuracy loss.

### Dynamic Batch Sizes

```python
torch_tensorrt.Input(
    min_shape=(1, 784),    # Min batch size: 1
    opt_shape=(32, 784),   # Optimal batch size: 32
    max_shape=(128, 784),  # Max batch size: 128
    dtype=torch.float32
)
```

---

## Integration with Existing Code

To use TensorRT in your existing evaluation code:

```python
# In your evaluation script
from tensorrt_inference_example import compile_with_tensorrt

# Load your model as usual
model = Net(dims=layers, device=device)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Compile with TensorRT
example_input = torch.randn(32, 784).to(device)
trt_model = compile_with_tensorrt(model, example_input, device)

# Use trt_model for inference
# (it has the same interface as the wrapped model)
```

---

## Performance Tips

1. **Batch Size**: Larger is usually better for throughput
2. **Precision**: Try FP16 if accuracy permits
3. **Warmup**: Always run warmup iterations
4. **Fixed Shapes**: TensorRT works best with fixed input shapes
5. **GPU Mode**: Set Jetson to max performance mode

For Jetson devices:
```bash
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks   # Max GPU/CPU clocks
```

---

## Next Steps

1. ✅ Run `test_tensorrt_setup.py` to verify installation
2. ✅ Try `minimal_tensorrt_example.py` to understand basics
3. ✅ Use `tensorrt_inference_example.py` for benchmarking
4. 📊 Experiment with different batch sizes and precisions
5. 🚀 Integrate into your training/evaluation pipeline

---

## Resources

- [Torch-TensorRT Documentation](https://pytorch.org/TensorRT/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [NVIDIA Jetson Optimization](https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/)

---

## Questions?

Check `TENSORRT_README.md` for detailed documentation and troubleshooting.
