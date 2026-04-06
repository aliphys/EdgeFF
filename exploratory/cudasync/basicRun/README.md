# basicRun Package Overview

This folder contains the core implementation and orchestration layer for the `basicRun` project.

## Structure

- `basicRun/`
  - `__init__.py` — package entrypoint for importing shared functionality.
  - `config.py` — shared YAML config loader with common defaults merge.
  - `data.py` — dataset loader helpers and transform definitions.
  - `model.py` — Forward-Forward network definition and training utilities.
  - `evaluation.py` — evaluation and inference helper functions.
  - `monitor.py` — Jetson hardware monitoring helpers.
  - `utils.py` — analysis utilities and helper functions.
- `configs/`
  - `common.yaml` — shared default values for all configs.
  - `run.yaml` — single training run configuration.
  - `sweep.yaml` — W&B sweep manifest.
  - `eval.yaml` — inference evaluation config.
  - `analysis.yaml` — visualization and analysis config.
- `scripts/`
  - `train.py` — wrapper entrypoint for training using `configs/run.yaml`.
  - `evaluate.py` — wrapper entrypoint supporting multiple evaluation modes.
  - `sweep.py` — wrapper entrypoint for launching W&B sweeps.
  - `legacy/` — original script implementations retained for compatibility.
- `tests/`
  - smoke and path tests for config and script structure.

## How it works

### Config loading

All YAML configs are loaded through `basicRun.config.load_config()`.
The loader merges the selected config file with `configs/common.yaml`, so shared values like `project`, `data_root`, and `hw_interval_ms` are defined once.

### Data and dataset handling

`basicRun.data` provides dataset transforms and loaders for:
- `MNIST`
- `FMNIST`
- `SVHN`
- `CIFAR10`

It returns both the dataset loader and whether the dataset is color.

### Model implementation

`basicRun.model` contains the Forward-Forward network:
- `Layer` — local layer trained by a goodness objective.
- `SoftmaxLayer` — classification head for each layer.
- `Net` — assembles the FF network and provides inference methods.

Training is performed by alternating representation-layer training and softmax-layer training.

### Evaluation

`basicRun.evaluation` contains generic evaluation helpers:
- `eval_train_set`
- `eval_test_set`
- `eval_val_set`
- `eval_val_set_light`
- `eval_with_inference_measurement`

The inference measurement helper can integrate with Jetson hardware monitoring when available.

### Hardware monitoring

`basicRun.monitor` provides Jetson-specific monitoring:
- `INA3221PowerMonitor` for power readings.
- `TegratsMonitor` for tegrastats and inference power sampling.
- `InferenceMetrics` for latency/energy metrics.

### Scripts

The top-level script wrappers keep the command surface small:
- `scripts/train.py` — trains or evaluates a model using `configs/run.yaml`.
- `scripts/evaluate.py` — runs evaluation subcommands (`inference`, `trt`, `energy`, `variance`, `analysis`).
- `scripts/sweep.py` — creates and optionally runs a W&B sweep using `configs/sweep.yaml`.

The original detailed scripts are kept in `scripts/legacy/` for compatibility but are no longer the main entrypoints.

## How to use

From `basicRun/`:

```bash
python scripts/train.py --config configs/run.yaml
python scripts/evaluate.py inference --config configs/eval.yaml
python scripts/evaluate.py analysis --config configs/analysis.yaml
python scripts/sweep.py --config configs/sweep.yaml --run-agent
```

## Testing

Run the smoke tests in `tests/`:

```bash
python tests/test_new_config_files.py
python tests/test_script_consolidation.py
```

## Notes

- The package is intentionally structured so core logic lives in `basicRun/` and scripts remain thin wrappers.
- `configs/common.yaml` centralizes shared defaults to minimize repetition.
- Legacy scripts are preserved under `scripts/legacy/` and can be migrated later if needed.
