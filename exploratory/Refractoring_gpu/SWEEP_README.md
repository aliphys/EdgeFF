# W&B Sweep Guide for Forward-Forward MNIST

This directory contains a configured W&B sweep for running grid search experiments on the Forward-Forward algorithm with MNIST.

## Sweep Configuration

The sweep is configured in `sweep_config.yaml` with the following parameters:

### Fixed Parameters
- **seed**: 42 (for reproducibility)
- **rep-epochs**: 3
- **softmax-epochs**: 3
- **train-batch-size**: 256

### Grid Search Parameters
- **no-cuda**: `[false, true]` - Test both GPU and CPU execution
- **layers**: 6 different network architectures:
  - `784,100,10` - 1 hidden layer, width 100
  - `784,100,100,10` - 2 hidden layers, width 100
  - `784,100,100,100,10` - 3 hidden layers, width 100
  - `784,200,10` - 1 hidden layer, width 200
  - `784,200,200,10` - 2 hidden layers, width 200
  - `784,200,200,200,10` - 3 hidden layers, width 200

**Total combinations**: 2 (cuda) × 6 (architectures) = **12 runs**

## How to Run the Sweep

### Option 1: Quick Start (Create and Run)
```bash
cd /home/jetson/Documents/github/EdgeFF/exploratory/Refractoring_gpu
python run_sweep.py --run-agent
```

### Option 2: Create Sweep, Then Run Manually
```bash
# Step 1: Create the sweep
python run_sweep.py

# This will output a sweep ID like: wandb agent <sweep-id>
# Step 2: Run the sweep agent
wandb agent <sweep-id>
```

### Option 3: Run Limited Number of Experiments
```bash
# Create and run only 3 experiments
python run_sweep.py --run-agent --count 3
```

### Option 4: Parallel Execution (Multiple Agents)
You can run multiple agents in parallel to speed up the sweep:

```bash
# Terminal 1
wandb agent <sweep-id>

# Terminal 2 (on the same or different machine)
wandb agent <sweep-id>

# Terminal 3
wandb agent <sweep-id>
```

Each agent will pick up the next available configuration from the sweep queue.

## Customizing the Sweep

### Edit the Configuration
Modify `sweep_config.yaml` to change parameters:

```yaml
parameters:
  no-cuda:
    values: [false, true]
  layers:
    values:
      - "784,100,10"
      - "784,500,500,10"  # Add new architecture
  rep-epochs:
    values: [3, 5, 10]  # Try multiple values
  seed:
    value: 42  # Or use: values: [42, 123, 456]
```

### Change Sweep Method
In `sweep_config.yaml`, you can change the method:

- **Grid search** (current): Tests all combinations
  ```yaml
  method: grid
  ```

- **Random search**: Randomly samples combinations
  ```yaml
  method: random
  ```

- **Bayesian optimization**: Smart parameter search
  ```yaml
  method: bayes
  ```

## Viewing Results

After running the sweep, you can view results at:
```
https://wandb.ai/<your-username>/edgeff/sweeps/<sweep-id>
```

The sweep will track:
- `final/test_accuracy` (optimization target)
- Training curves
- Hardware metrics (if available)
- Device information (CPU vs GPU)

## Expected Outputs

Each run will log:
- **Training metrics**: epoch-by-epoch validation accuracy
- **Final metrics**: test accuracy, train/val accuracy samples
- **Hardware metrics**: Power consumption, GPU utilization (if available)
- **Model artifacts**: Saved model checkpoints
- **Source code**: Logged as artifacts for reproducibility

## Troubleshooting

### Issue: "wandb: ERROR Error uploading"
**Solution**: Check your internet connection and W&B API key:
```bash
wandb login
```

### Issue: Out of memory (GPU)
**Solution**: Reduce batch size in `sweep_config.yaml`:
```yaml
parameters:
  train-batch-size:
    value: 128  # Reduced from 256
```

### Issue: Sweep runs too slowly
**Solution**: Run multiple agents in parallel (see Option 4 above)

## Single Run Testing

To test a single configuration before running the full sweep:
```bash
python Main.py --layers "784,100,10" --seed 42 --no-cuda
python Main.py --layers "784,200,200,10" --seed 42  # GPU version
```

## Project Structure
```
Refractoring_gpu/
├── Main.py                 # Main training script (sweep-compatible)
├── Train.py               # Training logic
├── Evaluation.py          # Evaluation functions
├── tools.py               # Utility functions
├── tegrats_monitor.py     # Hardware monitoring
├── sweep_config.yaml      # Sweep configuration
├── run_sweep.py          # Sweep launcher script
├── SWEEP_README.md       # This file
└── model/                # Saved model checkpoints
```
