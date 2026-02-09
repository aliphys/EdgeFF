# Refractoring_gpu - Forward-Forward Neural Network Training Framework

A GPU-optimized implementation of the Forward-Forward (FF) algorithm for training neural networks without backpropagation. This framework supports multiple datasets (MNIST, Fashion-MNIST, SVHN, CIFAR10), integrates with Weights & Biases for experiment tracking, and includes hardware monitoring for NVIDIA Jetson edge devices.

## Quick Start

```bash
# Single training run
python Main.py --layers "784,200,200,10" --rep-epochs 10 --softmax-epochs 10 --dataset MNIST --seed 42

# Run W&B hyperparameter sweep
python run_sweep.py --run-agent

# Evaluate trained models for inference metrics
python eval.py --config eval_config.yaml

# Variance analysis with GPU/CPU comparison (violin plots, throughput)
python eval_variance.py --config eval_variance_config.yaml

# Generate visualization plots
python analysis.py --config analysis_config.yaml
```

## Project Structure

| File | Purpose |
|------|---------|
| [Main.py](Main.py) | **Entry point** - Training orchestrator with CLI arguments and W&B sweep support |
| [Train.py](Train.py) | Core neural network classes (`Net`, `Layer`, `SoftmaxLayer`) and training logic |
| [Evaluation.py](Evaluation.py) | Model evaluation functions for train/val/test sets |
| [tools.py](tools.py) | Analysis utilities for confidence/goodness distributions |
| [tegrats_monitor.py](tegrats_monitor.py) | Hardware monitoring for Jetson devices (power, temperature, GPU%) |
| [eval.py](eval.py) | Batch inference evaluation of models from W&B |
| [eval_variance.py](eval_variance.py) | **Variance analysis** - GPU/CPU comparison, violin plots, throughput curves |
| [analysis.py](analysis.py) | Visualization and heatmap generation from W&B data |
| [run_sweep.py](run_sweep.py) | W&B sweep launcher script |
| [test_gpu_setup.py](test_gpu_setup.py) | GPU environment verification utility |
| [test_sweep_config.py](test_sweep_config.py) | Sweep configuration validation |

### Configuration Files

| File | Purpose |
|------|---------|
| [sweep_config.yaml](sweep_config.yaml) | Full grid search config (multi-dataset, seeds) |
| [sweep_width_config.yaml](sweep_width_config.yaml) | Width-focused sweep config |
| [eval_config.yaml](eval_config.yaml) | Inference evaluation settings |
| [eval_variance_config.yaml](eval_variance_config.yaml) | Variance analysis settings (GPU/CPU, batch sizes) |
| [analysis_config.yaml](analysis_config.yaml) | Visualization/analysis settings |

### Output Directories

| Directory | Contents |
|-----------|----------|
| `model/` | Saved model checkpoints |
| `wandb/` | Local W&B run data |
| `*.png` | Generated visualization plots |

## The Forward-Forward Algorithm

Unlike traditional backpropagation, the Forward-Forward algorithm trains each layer locally using a "goodness" objective:

1. **Positive samples**: Input images with correct labels embedded → maximize goodness
2. **Negative samples**: Input images with wrong labels embedded → minimize goodness
3. **Goodness function**: `mean(activations²)` per sample

Labels are embedded by replacing the first 10 pixels of the input with a one-hot encoding.

## Dependencies

```
torch, torchvision
wandb
matplotlib, pandas, numpy
scikit-learn
python-dotenv
pyyaml
```

For Jetson hardware monitoring:
- `tegrastats` CLI tool
- INA3221 sysfs interface

## Environment Setup

1. Create a `.env` file in the project root with your W&B API key:
   ```
   WANDB_API_KEY=your_key_here
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision wandb matplotlib pandas scikit-learn python-dotenv pyyaml
   ```

## CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--layers` | `"784,100,100,100"` | Comma-separated layer sizes (784 for MNIST, 3072 for color) |
| `--rep-epochs` | 10 | Representation training epochs |
| `--softmax-epochs` | 10 | Softmax classifier training epochs |
| `--train-batch-size` | 256 | Training mini-batch size |
| `--test-batch-size` | 512 | Validation/test batch size |
| `--dataset` | `MNIST` | Dataset: MNIST, FMNIST, SVHN, CIFAR10 |
| `--seed` | 0 | Random seed |
| `--no-cuda` | false | Disable CUDA |
| `--enable-hw-monitor` | true | Enable Jetson hardware monitoring |
| `--project` | `edgeff` | W&B project name |

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system architecture and data flow
- [LLMCONTEXT.md](LLMCONTEXT.md) - Compact reference for LLM context windows
- [SWEEP_README.md](SWEEP_README.md) - W&B sweep usage guide
- [GPU_CONVERSION_SUMMARY.md](GPU_CONVERSION_SUMMARY.md) - GPU optimization notes
