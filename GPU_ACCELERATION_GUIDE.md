# GPU Acceleration Implementation Guide for EdgeFF

## Overview
This document captures the complete implementation of CUDA GPU acceleration for the EdgeFF Forward-Forward training algorithm, including all lessons learned and common pitfalls to avoid.

## Performance Results
- **6.9x speedup** achieved on Jetson Orin GPU (7.4GB memory)
- CPU: 9.93 seconds vs GPU: 1.44 seconds (5K samples, 3 epochs)
- Maintains equivalent accuracy between CPU and GPU training

## Key Implementation Changes

### 1. Device Management Utilities Added to Train.py

```python
def get_device():
    """Get the best available device (CUDA if available, otherwise CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU (CUDA not available)")
    return device

def move_to_device(tensor, device):
    """Move tensor to specified device if it's not already there"""
    if tensor.device != device:
        return tensor.to(device)
    return tensor

def prepare_data_for_training(x_pos, x_neg, x_neutral, targets, device):
    """Move all training data to the specified device"""
    print(f"Moving training data to {device}...")
    x_pos = move_to_device(x_pos, device)
    x_neg = move_to_device(x_neg, device)
    x_neutral = move_to_device(x_neutral, device)
    targets = move_to_device(targets, device)
    print("✓ Data moved to device successfully")
    return x_pos, x_neg, x_neutral, targets
```

### 2. Updated Net Class Constructor

**CRITICAL CHANGE:** Added device parameter and proper ModuleList handling:

```python
class Net(torch.nn.Module):
    def __init__(self, dims, goodness_threshold=2.0, confidence_threshold_multiplier=1.0, device=None):
        super().__init__()
        self.layers = []
        self.softmax_layers = []
        self.confidence_threshold_multiplier = confidence_threshold_multiplier
        self.device = device if device is not None else torch.device('cpu')
        
        for d in range(len(dims) - 1):
            layer = Layer(dims[d], dims[d + 1], threshold=goodness_threshold)
            self.layers += [layer]
        for d in range(1, len(dims)):
            in_dim = dims[d]
            for i in range(1, d):
                in_dim += dims[i]
            softmax_layer = SoftmaxLayer(in_features=in_dim, out_features=10)
            self.softmax_layers += [softmax_layer]
        
        # CRITICAL: Convert to ModuleLists for proper device handling
        self.layers = torch.nn.ModuleList(self.layers)
        self.softmax_layers = torch.nn.ModuleList(self.softmax_layers)
```

### 3. Updated build_model Function

**MAJOR CHANGE:** Automatic GPU detection and data movement:

```python
def build_model(x_pos, x_neg, x_neutral, targets, layers, goodness_threshold=2.0, confidence_threshold_multiplier=1.0):
    dims = layers  
    
    # Set up device and move data to GPU
    device = get_device()
    x_pos, x_neg, x_neutral, targets = prepare_data_for_training(x_pos, x_neg, x_neutral, targets, device)
    
    # Create model and move to GPU
    model = Net(dims, goodness_threshold=goodness_threshold, confidence_threshold_multiplier=confidence_threshold_multiplier, device=device)
    model = model.to(device)
    print(f"✓ Model moved to {device}")
    
    # Continue with existing training logic...
```

## CRITICAL BUG FIXES - Lessons Learned

### ❌ Problem 1: CPU/GPU Tensor Mixing
**Error:** `Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0`

**Root Cause:** Original code had `.cpu()` calls that forced GPU tensors back to CPU during forward passes.

**❌ Original Buggy Code:**
```python
# BAD - Forces tensor back to CPU
softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
softmax_layer_input = h.cpu()
```

**✅ Fixed Code:**
```python
# GOOD - Keeps tensors on same device
softmax_layer_input = torch.cat((softmax_layer_input, h), 1)
softmax_layer_input = h
```

**Locations Fixed:**
- `predict_one_pass()` method
- `light_predict_one_sample()` method
- `light_predict_analysis()` method

### ❌ Problem 2: Tensor Creation on Wrong Device
**Error:** Device mismatch when creating new tensors during training.

**Root Cause:** `torch.empty()` creates tensors on CPU by default, even when input data is on GPU.

**❌ Original Buggy Code:**
```python
# BAD - Creates tensor on CPU regardless of input device
softmax_layer_input = torch.empty((batch_size, num_input_features))
```

**✅ Fixed Code:**
```python
# GOOD - Creates tensor on same device as input data
softmax_layer_input = torch.empty((batch_size, num_input_features), device=x_neutral_label.device)
```

**Location Fixed:** `train_softmax_layer()` method

### ❌ Problem 3: Improper Module Registration
**Error:** Layers not properly recognized by PyTorch's device management.

**Root Cause:** Using Python lists instead of `torch.nn.ModuleList` prevents proper device handling.

**❌ Original Buggy Code:**
```python
# BAD - Python lists don't register with PyTorch
self.layers = []
self.softmax_layers = []
# ... populate lists ...
```

**✅ Fixed Code:**
```python
# GOOD - ModuleLists properly register with PyTorch
self.layers = []
self.softmax_layers = []
# ... populate lists ...
# Convert to ModuleLists for proper device handling
self.layers = torch.nn.ModuleList(self.layers)
self.softmax_layers = torch.nn.ModuleList(self.softmax_layers)
```

## Testing Infrastructure Created

### 1. test_cuda.py
- Verifies CUDA availability and GPU information
- Tests basic tensor operations on GPU
- Performance comparison for matrix operations
- **Usage:** `python test_cuda.py`

### 2. test_gpu_training.py
- End-to-end training validation with small dataset
- Tests model creation and GPU movement
- Validates complete training pipeline
- **Usage:** `python test_gpu_training.py`

### 3. performance_test.py
- Comprehensive CPU vs GPU training comparison
- Uses realistic MNIST data and architecture
- Measures both performance and accuracy
- **Usage:** `python performance_test.py`

## GPU Acceleration Checklist

When implementing GPU acceleration in similar projects, ensure:

### ✅ Data Movement
- [ ] All input tensors moved to GPU before training
- [ ] Model moved to GPU using `.to(device)`
- [ ] No `.cpu()` calls in forward passes unless explicitly needed
- [ ] New tensors created with correct `device=` parameter

### ✅ Model Architecture
- [ ] Use `torch.nn.ModuleList` instead of Python lists
- [ ] Pass device parameter to model constructor
- [ ] Ensure all layers are properly registered as modules

### ✅ Training Loop
- [ ] Batch data moved to same device as model
- [ ] No unnecessary CPU↔GPU transfers in training loop
- [ ] Use `torch.cuda.synchronize()` for accurate timing

### ✅ Memory Management
- [ ] Monitor GPU memory usage
- [ ] Use appropriate batch sizes for GPU memory
- [ ] Clear gradients properly to prevent memory leaks

## Common Pitfalls to Avoid

1. **Mixed Device Operations:** Never mix CPU and GPU tensors in the same operation
2. **Automatic CPU Creation:** Always specify `device=` when creating new tensors
3. **Premature CPU Movement:** Avoid `.cpu()` calls during training unless necessary
4. **Module Registration:** Use ModuleList/ModuleDict for proper PyTorch integration
5. **Memory Allocation:** Don't assume GPU has unlimited memory - monitor usage

## Performance Optimization Tips

1. **Batch Size:** Increase batch sizes to better utilize GPU parallelism
2. **Data Loading:** Use `pin_memory=True` in DataLoader for faster CPU→GPU transfer
3. **Mixed Precision:** Consider using `torch.cuda.amp` for faster training with large models
4. **Memory Management:** Use `torch.cuda.empty_cache()` if experiencing memory issues

## Verification Commands

After any GPU-related changes, run these tests:

```bash
# Test CUDA setup
python test_cuda.py

# Test training pipeline
python test_gpu_training.py

# Performance comparison
python performance_test.py

# Full training (should automatically use GPU)
python Main.py
```

## Hardware Tested On

- **Platform:** NVIDIA Jetson Orin
- **GPU Memory:** 7.4 GB
- **CUDA Version:** 12.6
- **PyTorch Version:** 2.8.0
- **Compute Capability:** 8.7

## Future Enhancements

1. **Multi-GPU Support:** Extend to support multiple GPUs using DataParallel
2. **Memory Optimization:** Implement gradient checkpointing for larger models
3. **Mixed Precision:** Add automatic mixed precision training
4. **Profiling:** Add detailed GPU profiling and bottleneck analysis

---

**Last Updated:** September 24, 2025  
**Status:** Production Ready ✅  
**Performance Gain:** 6.9x speedup achieved
