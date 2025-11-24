# GPU Conversion Summary

This document summarizes the changes made to enable GPU acceleration for the Forward-Forward MNIST training and inference pipeline.

## Files Modified

### 1. `Train.py`
**Key Changes:**
- Added `device` parameter to `Net.__init__()` with auto-detection of GPU availability
- All layers (`Layer` and `SoftmaxLayer`) are moved to device during initialization
- Updated all methods to accept inputs on any device and move them to the model's device:
  - `predict_one_pass()`: Moves input to device, keeps all intermediate activations on GPU
  - `light_predict_one_sample()`: Moves input to device, processes on GPU
  - `light_predict_analysis()`: Moves input to device, only transfers to CPU for final numpy conversions
  - `train()`: Moves positive and negative samples to device
  - `train_softmax_layer()`: Moves inputs and targets to device, creates tensors directly on GPU
- Updated `build_model()` function to detect GPU and move all training data to device once

**Benefits:**
- Minimal data transfers between CPU and GPU
- All matrix operations execute on GPU
- Automatic fallback to CPU if GPU unavailable

### 2. `Main.py`
**Key Changes:**

#### Device Configuration (Lines ~52-61)
```python
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA even if available.')
args = parser.parse_args()

# Device configuration
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")
if use_cuda:
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

#### Wandb Config Update (Lines ~105-110)
```python
'enable_hw_monitor': args.enable_hw_monitor,
'hw_interval_ms': args.hw_interval_ms,
'device': str(device),
'cuda_available': torch.cuda.is_available(),
'cuda_device_name': torch.cuda.get_device_name(0) if use_cuda else None,
```

#### Model Initialization (Line ~209)
```python
model = Train.Net(layers, device=device)
```

#### Validation Subset on Device (Lines ~214-217)
```python
val_subset_inputs, val_subset_targets = next(iter(val_loader))
# Move to device
val_subset_inputs = val_subset_inputs.to(device)
val_subset_targets = val_subset_targets.to(device)
```

#### Training Loops - Data Movement (Lines ~222-236, ~252-256)
```python
# Representation training loop
for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
    # Move batch to device
    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)
    # ... rest of training code

# Softmax training loop
for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
    # Move batch to device
    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)
    # ... rest of training code
```

#### Model Loading with Device Support (Lines ~273-281)
```python
else:
    name = 'temp_'
    model = torch.load(os.path.split(os.path.realpath(__file__))[0] + '/model/' + name)
    # Ensure loaded model is on the correct device
    if hasattr(model, 'device'):
        model.device = device
        # Move all layers to device
        for layer in model.layers:
            layer.to(device)
        for softmax_layer in model.softmax_layers:
            softmax_layer.to(device)
```

#### Evaluation Data on Device (Lines ~284-300)
```python
# Train set
train_inputs_full = torch.cat([d for d, _ in train_loader], dim=0).to(device)
train_targets_full = torch.cat([t for _, t in train_loader], dim=0).to(device)

# Test set
test_inputs_full = torch.cat([d for d, _ in test_loader], dim=0).to(device)
test_targets_full = torch.cat([t for _, t in test_loader], dim=0).to(device)

# Validation set
val_inputs_full = torch.cat([d for d, _ in val_loader], dim=0).to(device)
val_targets_full = torch.cat([t for _, t in val_loader], dim=0).to(device)
```

### 3. `Evaluation.py`
**Status:** ✅ Already GPU-compatible
- All functions properly use `.detach().cpu().numpy()` when converting GPU tensors to numpy
- No changes needed

### 4. `tools.py`
**Status:** ✅ Already GPU-compatible
- `analysis_val_set()` properly handles GPU tensors with `.detach().cpu().numpy()`
- No changes needed

## Data Movement Strategy

### Principle: Move Once, Process on GPU, Transfer Back Only When Needed

1. **Training Phase:**
   - Each batch is moved to GPU once when loaded from DataLoader
   - All forward/backward passes happen on GPU
   - No intermediate CPU transfers

2. **Evaluation Phase:**
   - Full datasets concatenated and moved to GPU once
   - All predictions computed on GPU
   - Only final results transferred to CPU for metrics calculation

3. **Model Persistence:**
   - Loaded models automatically moved to the configured device
   - All layers explicitly transferred to maintain consistency

## Usage

### Run with GPU (default if available):
```bash
python Main.py --layers 784,100,100,100 --rep-epochs 3 --softmax-epochs 3
```

### Run without GPU (force CPU):
```bash
python Main.py --layers 784,100,100,100 --rep-epochs 3 --softmax-epochs 3 --no-cuda
```

### Check GPU utilization during training:
```bash
# In another terminal
watch -n 1 nvidia-smi
# or for Jetson devices
tegrastats
```

## Expected Performance Improvements

- **Training Speed:** 5-20x faster depending on GPU
- **Inference Speed:** 10-50x faster for batch processing
- **Memory:** More efficient use of GPU memory by keeping data on device

## Verification Checklist

- [x] Device auto-detection and configuration
- [x] Model layers moved to GPU
- [x] Training data moved to GPU in batches
- [x] Validation subset moved to GPU
- [x] Evaluation data moved to GPU
- [x] Model loading handles device placement
- [x] Wandb logs device information
- [x] All intermediate computations stay on GPU
- [x] Only final outputs transferred to CPU when needed
- [x] Backward compatibility with CPU-only systems

## Notes

- The `--no-cuda` flag allows forcing CPU execution even when GPU is available (useful for debugging or comparison)
- Device information is logged to wandb for experiment tracking
- The Train.py module uses `device=None` parameter with auto-detection as default
- All existing evaluation and analysis functions work seamlessly with GPU tensors
