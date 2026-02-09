# Architecture Documentation

This document describes the system architecture, class hierarchy, and data flow of the Forward-Forward training framework.

## File Import Relationships

```
Main.py (entry point)
  ├── Train.py ──────────────── Net, Layer, SoftmaxLayer, overlay functions
  ├── Evaluation.py ─────────── eval_train_set, eval_test_set, eval_val_set, eval_val_set_light
  ├── tools.py ──────────────── analysis_val_set, calculate_goodness_distributions
  └── tegrats_monitor.py ────── TegratsMonitor, INA3221PowerMonitor, InferenceMetrics

eval.py (batch inference)
  └── Evaluation.py

analysis.py (visualization)
  └── (standalone - reads from wandb API)

run_sweep.py (sweep launcher)
  └── (standalone - launches wandb agent)

tools.py
  └── Evaluation.py ─────────── print_results
```

## Class Hierarchy

```
torch.nn.Module
    │
    └── Net ─────────────────── Main network container
            │
            ├── layers: List[Layer]
            │       └── Layer (extends nn.Linear) ─── FF layer with goodness training
            │
            └── softmax_layers: List[SoftmaxLayer]
                    └── SoftmaxLayer (extends nn.Module) ─── Classification head
```

### Net Class

The main neural network container managing Forward-Forward layers and softmax classifiers.

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `predict_one_pass(x, batch_size)` | Full inference through all layers |
| `light_predict_one_sample(x, mean, std)` | Early-exit inference based on confidence thresholds |
| `light_predict_analysis(x, num_layers)` | Analysis mode returning per-layer predictions |
| `train(x_pos, x_neg)` | Train representation layers with positive/negative samples |
| `train_softmax_layer(x, y, batch_size, dims)` | Train softmax classification heads |
| `check_confidence(layer, mean, std, output)` | Check if early exit is confident |

### Layer Class (extends nn.Linear)

A single Forward-Forward layer with ReLU activation and threshold-based training.

**Key Features:**
- L2 norm direction normalization before linear transformation
- Goodness function: `mean(h²)` per sample
- Loss: `log(1 + exp(-(goodness - threshold)))` for positives, `log(1 + exp(goodness - threshold))` for negatives

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `forward(x)` | ReLU(Linear(normalize(x))) |
| `train(x_pos, x_neg)` | Single FF training step |

### SoftmaxLayer Class

Classification head using softmax with cross-entropy loss.

**Key Features:**
- Receives concatenated activations from all previous layers
- Standard cross-entropy training

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `forward(x)` | Returns (logits, softmax_output) |
| `train(x, y)` | Single cross-entropy training step |

## Data Flow

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA LOADING                                   │
│   dataset_loaders() → MNIST/FMNIST/SVHN/CIFAR10                         │
│   → Normalize → Flatten → DataLoader (train/val/test splits)            │
│                                                                          │
│   Input dimensions: 784 (grayscale) or 3072 (RGB)                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: REPRESENTATION TRAINING                      │
│                                                                          │
│   For each epoch:                                                        │
│     For each batch (x, y):                                               │
│       x_pos = overlay_y_on_x(x, y_true)    ← Correct label embedded     │
│       y_neg = random wrong labels                                        │
│       x_neg = overlay_y_on_x(x, y_neg)     ← Wrong label embedded       │
│       model.train(x_pos, x_neg)            ← FF goodness objective      │
│                                                                          │
│   Each layer trains independently (no backprop through network)          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: SOFTMAX LAYER TRAINING                       │
│                                                                          │
│   For each epoch:                                                        │
│     For each batch (x, y):                                               │
│       x_neutral = overlay_on_x_neutral(x)  ← 0.1 values (no label)      │
│       model.train_softmax_layer(x_neutral, y, ...)                       │
│                                                                          │
│   Softmax layer d receives concatenated activations from layers 1..d    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION                                       │
│                                                                          │
│   Standard Inference:                                                    │
│     eval_train_set() / eval_test_set() / eval_val_set()                 │
│     → model.predict_one_pass() → accuracy, F1-score                     │
│                                                                          │
│   Early-Exit Inference:                                                  │
│     analysis_val_set() → compute confidence thresholds (mean ± std)     │
│     eval_val_set_light() → dynamic exit, returns layers used            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LOGGING & ARTIFACTS                                 │
│                                                                          │
│   wandb.log() → metrics per epoch:                                       │
│     - sample_val_accuracy                                                │
│     - final/test_accuracy, final/train_accuracy_sample                   │
│     - Hardware metrics (power, GPU%, RAM, temperature)                   │
│                                                                          │
│   wandb.log_artifact() → model checkpoint, source code                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Network Forward Pass (Inference)

```
Input (784 or 3072 pixels)
    │
    ▼ [Neutral label: first 10 pixels set to 0.1]
┌───────────────────────────────────────────────────────────────┐
│ Layer 1: normalize(x) → Linear(in, hidden) → ReLU → h₁       │
├───────────────────────────────────────────────────────────────┤
│ SoftmaxLayer 1: h₁ → Linear(hidden, 10) → softmax            │
│                 Can classify after first layer                │
└───────────────────────────────────────────────────────────────┘
    │ h₁
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Layer 2: normalize(h₁) → Linear(hidden, hidden) → ReLU → h₂  │
├───────────────────────────────────────────────────────────────┤
│ SoftmaxLayer 2: [h₁ ‖ h₂] → Linear(2×hidden, 10) → softmax   │
│                 Cumulative activations from all prior layers  │
└───────────────────────────────────────────────────────────────┘
    │ h₂
    ▼
   ... (more layers)
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Final SoftmaxLayer: [h₁ ‖ h₂ ‖ ... ‖ hₙ] → prediction        │
└───────────────────────────────────────────────────────────────┘
```

### Early Exit Decision

```python
# For each layer i:
threshold = confidence_mean[i] - confidence_std[i]
if max(softmax_output) > threshold:
    return prediction  # Exit early
# Otherwise continue to next layer
```

## Label Embedding

### Grayscale Images (MNIST, Fashion-MNIST)
- First 10 pixels replaced with one-hot encoding
- `x[:, :10] = 0; x[:, label] = 10.0`

### Color Images (SVHN, CIFAR10)
- First 10 pixels of EACH channel replaced
- Red: pixels 0-9, Green: pixels 1024-1033, Blue: pixels 2048-2057
- `x[:, channel*1024 + :10] = 0; x[:, channel*1024 + label] = 10.0`

### Neutral Embedding (for inference)
- First 10 pixels set to 0.1 (no label information)
- Used during softmax training and inference

## Hardware Monitoring (Jetson)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TegratsMonitor                                    │
│   - Parses tegrastats CLI output                                        │
│   - Captures: RAM, GPU%, CPU%, temperature                              │
│   - Runs in background thread                                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     INA3221PowerMonitor                                  │
│   - Reads from /sys/bus/i2c/drivers/ina3221/...                         │
│   - Channels:                                                            │
│       1: VDD_IN (Total Module Power)                                    │
│       2: VDD_CPU_GPU_CV (CPU + GPU + CV cores)                          │
│       3: VDD_SOC (Memory subsystem, nvdec, nvenc, etc.)                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      InferenceMetrics                                    │
│   - Brackets inference with start_inference() / end_inference()         │
│   - Collects power samples during inference                              │
│   - Calculates: energy_per_sample_mj, latency_per_sample_ms            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Configuration Schema

### Sweep Config (sweep_config.yaml)
```yaml
program: Main.py
method: grid|random|bayes
metric:
  name: final/test_accuracy
  goal: maximize
parameters:
  layers:
    values: ["784,100,100,10", "784,500,500,10", ...]
  dataset:
    values: [MNIST, FMNIST, SVHN, CIFAR10]
  seed:
    values: [42, 123, 456]
  no_cuda:
    values: [false, true]
```

### Eval Config (eval_config.yaml)
```yaml
project: edgeff-network-width
sweep_id: <sweep_id>             # Source sweep for trained models
dataset: MNIST
inference_batch_sizes: [1, 2, 8, 16, 32, 64, 128, 256, 512]
hw_interval_ms: 500
```

### Analysis Config (analysis_config.yaml)
```yaml
project: edgeff-network-width
sweep_id: <sweep_id>             # Source sweep for analysis
```
