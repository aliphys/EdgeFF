#!/usr/bin/env python3

import sys
from pathlib import Path

# Ensure the parent folder containing basicRun is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))

import torch
from basicRun.config import load_config
from basicRun.data import get_test_loader
from basicRun.model import Net

def main():
    print("=== basicRun smoke test ===")

    config_path = ROOT.parent / "configs" / "eval_config.yaml"
    config = load_config(config_path)
    print("Loaded config project:", config.get("project"))
    print("Loaded sweep_id:", config.get("sweep_id"))

    loader, is_color = get_test_loader("MNIST", batch_size=2, data_root=ROOT / "data")
    x_batch, y_batch = next(iter(loader))
    print("MNIST batch shape:", x_batch.shape, "labels shape:", y_batch.shape)
    print("is_color:", is_color)

    model = Net([784, 100, 10], device=torch.device("cpu"), is_color=False)
    preds = model.predict_one_pass(x_batch, batch_size=2)
    print("Model prediction shape:", preds.shape)
    print("Predictions:", preds.tolist())

    if preds.shape == (2,):
        print("Smoke test PASSED")
    else:
        raise RuntimeError("Smoke test failed: unexpected prediction shape")

if __name__ == "__main__":
    main()