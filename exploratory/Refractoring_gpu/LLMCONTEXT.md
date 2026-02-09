# LLM Context Reference - Refractoring_gpu

Compact reference for LLM context windows. This file contains essential information for understanding and modifying this codebase.

## Purpose

GPU-optimized Forward-Forward (FF) neural network training without backpropagation. Supports MNIST, Fashion-MNIST, SVHN, CIFAR10. Integrates with W&B for experiment tracking.

## File Quick Reference

| File | What it does |
|------|--------------|
| `Main.py` | Entry point. CLI args → data loading → training → evaluation → W&B logging |
| `Train.py` | `Net`, `Layer`, `SoftmaxLayer` classes. FF training logic. |
| `Evaluation.py` | `eval_train_set()`, `eval_test_set()`, `eval_val_set()`, `eval_val_set_light()` |
| `tools.py` | `analysis_val_set()` - compute confidence thresholds for early exit |
| `tegrats_monitor.py` | Jetson power/temperature monitoring |
| `eval.py` | Batch inference evaluation from W&B sweep |
| `analysis.py` | Generate plots from W&B data |
| `run_sweep.py` | Launch W&B hyperparameter sweeps |

## Core Classes

### `Net` (Train.py)
```python
class Net(torch.nn.Module):
    def __init__(self, dims, device=None, onehot_max_value=10.0, is_color=False)
    # dims: [784, 100, 100, 10] or [3072, 500, 500, 10]
    
    def train(self, x_pos, x_neg)  # FF representation training
    def train_softmax_layer(self, x_neutral, y, batch_size, dims)  # softmax training
    def predict_one_pass(self, x, batch_size)  # full inference → predictions
    def light_predict_one_sample(self, x, mean, std)  # early-exit inference
    def light_predict_analysis(self, x, num_layers)  # returns per-layer predictions
```

### `Layer` (Train.py, extends nn.Linear)
```python
class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None)
    def forward(self, x)  # ReLU(Linear(normalize(x)))
    def train(self, x_pos, x_neg)  # single FF training step
    # Goodness = mean(activations²)
    # Loss: log(1 + exp(±(goodness - threshold)))
```

### `SoftmaxLayer` (Train.py)
```python
class SoftmaxLayer(nn.Module):
    def __init__(self, in_features, out_features)
    def forward(self, x)  # returns (logits, softmax_output)
    def train(self, x, y)  # cross-entropy training step
```

## Key Functions

### Label Embedding (Train.py)
```python
overlay_y_on_x(x, y, max_value=10.0, is_color=False)
# Embeds one-hot label in first 10 pixels (per channel for color)
# Used for positive (correct label) and negative (wrong label) samples

overlay_on_x_neutral(x, is_color=False)
# Sets first 10 pixels to 0.1 (neutral, no label info)
# Used during inference and softmax training
```

### Evaluation (Evaluation.py)
```python
eval_train_set(model, inputs, targets)  # prints accuracy/F1 on train
eval_test_set(model, inputs, targets)   # prints accuracy/F1 on test
eval_val_set(model, inputs, targets)    # prints accuracy/F1 on validation
eval_val_set_light(model, inputs, targets, mean, std)  # early-exit evaluation
eval_val_set_with_energy(model, inputs, targets, power_monitor, inference_metrics)
```

### Analysis (tools.py)
```python
analysis_val_set(model, inputs, targets)
# Returns: (mean_per_layer, std_per_layer) for confidence thresholds
# Used to set early-exit thresholds: threshold[i] = mean[i] - std[i]
```

## Training Flow

```
1. Load data → train_loader, val_loader, test_loader
2. Create model: Net(dims=[784,100,100,10], device=cuda)
3. Phase 1 - Representation:
   for epoch in rep_epochs:
     for x, y in train_loader:
       x_pos = overlay_y_on_x(x, y)        # correct label
       x_neg = overlay_y_on_x(x, random_y) # wrong label
       model.train(x_pos, x_neg)
4. Phase 2 - Softmax:
   for epoch in softmax_epochs:
     for x, y in train_loader:
       x_neutral = overlay_on_x_neutral(x)
       model.train_softmax_layer(x_neutral, y, batch_size, dims)
5. Evaluate: eval_test_set(model, test_inputs, test_targets)
```

## CLI Arguments (Main.py)

```
--layers "784,100,100,10"  # network architecture
--rep-epochs 10            # representation training epochs
--softmax-epochs 10        # softmax training epochs
--train-batch-size 256     # training batch size
--dataset MNIST            # MNIST|FMNIST|SVHN|CIFAR10
--seed 42                  # random seed
--no-cuda false            # disable GPU
--project edgeff           # wandb project name
```

## Input Dimensions

| Dataset | Input dim | is_color |
|---------|-----------|----------|
| MNIST | 784 | False |
| FMNIST | 784 | False |
| SVHN | 3072 | True |
| CIFAR10 | 3072 | True |

## Common Modifications

### Add a new dataset
1. In `Main.py`: Add to `dataset_loaders()` with transform and normalization
2. Set correct `is_color` flag and `onehot_max_value`
3. Update `expected_input_dim` validation

### Change network architecture
```bash
python Main.py --layers "784,500,500,500,10"  # 3 hidden layers of 500
```

### Add new metric to W&B
```python
# In Main.py, after training:
wandb.log({"my_metric": value})
```

### Modify early-exit threshold
```python
# In Net.check_confidence():
threshold = confidence_mean_vec[layer_num] - confidence_std_vec[layer_num]
# Change to e.g.: threshold = confidence_mean_vec[layer_num] - 2*confidence_std_vec[layer_num]
```

### Change goodness function
```python
# In Layer.train():
g_pos = h_pos.pow(2).mean(1)  # current: mean squared activations
# Change to e.g.: g_pos = h_pos.abs().mean(1)  # mean absolute activations
```

## W&B Integration

- Project set via `--project` argument
- Metrics logged: `sample_val_accuracy`, `final/test_accuracy`, hardware metrics
- Artifacts: model checkpoints (`.pth` files), source code
- Sweeps: grid/random/bayes over hyperparameters

## Hardware Monitoring (Jetson only)

```python
from tegrats_monitor import TegratsMonitor, INA3221PowerMonitor

# Start background monitoring
monitor = TegratsMonitor(interval_ms=500)
monitor.start()

# Get power readings
power_monitor = INA3221PowerMonitor()
metrics = power_monitor.get_power_metrics()
# Returns: VDD_IN_power_mw, VDD_CPU_GPU_CV_power_mw, VDD_SOC_power_mw, etc.
```
