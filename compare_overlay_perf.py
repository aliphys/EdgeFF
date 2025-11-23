"""
Benchmark overlay_y_on_x implementations on MNIST test set and plot results.

Two versions compared:
- baseline_overlay: manual indexing version (zero first K features and set x[batch_idx, y] = x.max())
- onehot_overlay: uses torch.nn.functional.one_hot to form one-hot and assign in a vectorized way

The script loads the MNIST test dataset, flattens to (N, 784), and measures
execution time on CPU and, if available, CUDA. It verifies both methods produce
identical outputs and saves a bar chart plot under output/.

Usage (optional): simply run `python compare_overlay_perf.py`.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

import matplotlib
matplotlib.use("Agg")  # non-interactive backend suitable for headless environments
import matplotlib.pyplot as plt


def baseline_overlay(x: torch.Tensor, y: torch.Tensor, classes: int = 10) -> torch.Tensor:
    """Original manual indexing version.

    x: (B, F) float tensor
    y: (B,) long tensor with class indices in [0, classes)
    returns: (B, F)
    """
    x_ = x.clone()
    x_[:, :classes] *= 0.0
    # Ensure y is long for indexing
    y_idx = y.to(dtype=torch.long)
    x_[torch.arange(x.shape[0], device=x.device), y_idx] = x.max()
    return x_


def onehot_overlay(x: torch.Tensor, y: torch.Tensor, classes: int = 10) -> torch.Tensor:
    """One-hot based version using torch.nn.functional.one_hot.

    x: (B, F) float tensor
    y: (B,) long tensor
    returns: (B, F)
    """
    x_ = x.clone()
    onehot = F.one_hot(y.to(dtype=torch.long), num_classes=classes).to(dtype=x.dtype, device=x.device)
    x_[:, :classes] = onehot * x.max()
    return x_


@torch.inference_mode()
def load_mnist_test(flatten: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST test set as tensors (X, y).

    If torchvision is unavailable or dataset load fails, falls back to synthetic
    data with the same shapes as MNIST test set (10000 x 784).
    """
    data_root = os.path.join(os.path.dirname(__file__), "data")
    if _HAS_TORCHVISION:
        try:
            transform = transforms.ToTensor()
            test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
            loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, num_workers=0)
            (imgs, labels) = next(iter(loader))
            # imgs: (N, 1, 28, 28)
            if flatten:
                x = imgs.view(imgs.shape[0], -1).contiguous()
            else:
                x = imgs
            y = labels.to(torch.long)
            return x, y
        except Exception as e:
            print(f"Warning: Failed to load MNIST via torchvision ({e}). Falling back to synthetic data.")

    # Fallback synthetic data
    N = 10_000
    Fdim = 28 * 28
    x = torch.rand(N, Fdim)
    y = torch.randint(low=0, high=10, size=(N,), dtype=torch.long)
    return x, y


def _time_once(fn, x, y, classes, device: torch.device) -> float:
    """Time a single call of fn(x, y, classes) on the given device."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = fn(x, y, classes)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0


def benchmark_on_device(x_cpu: torch.Tensor, y_cpu: torch.Tensor, device: torch.device, classes: int, repeats: int) -> Dict[str, float]:
    """Benchmark both methods on the specified device and return avg times in seconds."""
    x = x_cpu.to(device, non_blocking=True)
    y = y_cpu.to(device, non_blocking=True)

    # Warm-up
    for _ in range(3):
        baseline_overlay(x, y, classes)
        onehot_overlay(x, y, classes)

    if device.type == "cuda":
        torch.cuda.synchronize()

    base_times: List[float] = []
    onehot_times: List[float] = []
    for _ in range(repeats):
        base_times.append(_time_once(baseline_overlay, x, y, classes, device))
        onehot_times.append(_time_once(onehot_overlay, x, y, classes, device))

    # Correctness check (once)
    base_out = baseline_overlay(x, y, classes)
    onehot_out = onehot_overlay(x, y, classes)
    equal = torch.allclose(base_out, onehot_out)
    if not equal:
        max_diff = (base_out - onehot_out).abs().max().item()
        raise AssertionError(f"Outputs differ on device {device}! max_diff={max_diff}")

    return {
        "baseline_avg_s": sum(base_times) / len(base_times),
        "onehot_avg_s": sum(onehot_times) / len(onehot_times),
    }


def plot_results(results: Dict[str, Dict[str, float]], save_dir: str) -> str:
    """Create a bar chart of average times per device and return saved path."""
    devices = list(results.keys())
    baseline_vals = [results[d]["baseline_avg_s"] for d in devices]
    onehot_vals = [results[d]["onehot_avg_s"] for d in devices]

    x_locs = range(len(devices))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x_locs], baseline_vals, width=width, label="baseline")
    plt.bar([i + width / 2 for i in x_locs], onehot_vals, width=width, label="one_hot")
    plt.xticks(list(x_locs), devices)
    plt.ylabel("Average time (s)")
    plt.title("overlay_y_on_x: baseline vs one-hot")
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"overlay_overlay_perf_{ts}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    torch.manual_seed(0)
    classes = 10
    repeats = int(os.environ.get("OVERLAY_BENCH_REPEATS", "50"))

    print("Loading MNIST test set (or synthetic fallback)…")
    x_cpu, y_cpu = load_mnist_test(flatten=True)
    # Ensure float32 for x
    x_cpu = x_cpu.to(torch.float32)
    print(f"Dataset: x={tuple(x_cpu.shape)} dtype={x_cpu.dtype}, y={tuple(y_cpu.shape)}")
    print(f"x.min={x_cpu.min().item():.4f}, x.max={x_cpu.max().item():.4f}")

    devices: List[torch.device] = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    results: Dict[str, Dict[str, float]] = {}
    for dev in devices:
        print(f"\nBenchmarking on {dev} with repeats={repeats}…")
        res = benchmark_on_device(x_cpu, y_cpu, dev, classes=classes, repeats=repeats)
        results[dev.type] = res
        speedup = res["baseline_avg_s"] / res["onehot_avg_s"] if res["onehot_avg_s"] > 0 else float("inf")
        print(
            f"{dev}: baseline={res['baseline_avg_s']*1e3:.3f} ms | one_hot={res['onehot_avg_s']*1e3:.3f} ms | "
            f"speedup={speedup:.2f}x"
        )

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    out_path = plot_results(results, out_dir)
    print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    main()
