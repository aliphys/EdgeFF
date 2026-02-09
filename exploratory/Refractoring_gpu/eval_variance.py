"""
eval_variance.py - Inference Variance Analysis with GPU/CPU Comparison + Energy
================================================================================

This script evaluates trained models from a W&B sweep multiple times to
capture inference latency/energy variance, throughput metrics, and GPU vs CPU
comparison with violin plots.

Features:
    - Runs inference on both CUDA and CPU for comparison
    - Multiple batch sizes for throughput analysis
    - Per-sample latency tracking (batch_size=1)
    - Energy measurement on Jetson devices (INA3221 power sensor)
    - Warmup run discarded for each batch size (fresh timing)
    - Model loaded fresh for each device (no transfer bias)

Usage:
    python eval_variance.py --config eval_variance_config.yaml
    python eval_variance.py --sweep-id 9xqabsi3 --iterations 50

Output:
    - violin_latency_by_width.png: Latency distribution per width (GPU vs CPU)
    - violin_energy_by_width.png: Energy distribution per width (Jetson only)
    - throughput_vs_batchsize.png: Throughput curves by batch size
    - energy_vs_batchsize.png: Energy per sample curves by batch size
    - speedup_heatmap.png: GPU speedup ratio heatmap
    - energy_heatmap.png: Energy consumption heatmap
    - per_sample_histogram.png: Per-sample latency histogram
    - latency_stats.csv: Statistics per (width, batch_size, device)
    - W&B logging of all measurements
"""

import torch
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import wandb
import argparse
import yaml
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict


# Try to import hardware monitoring (Jetson only)
HW_MONITOR_AVAILABLE = False
INA3221PowerMonitor = None
TegratsMonitor = None
try:
    from tegrats_monitor import INA3221PowerMonitor, TegratsMonitor, InferenceMetrics
    HW_MONITOR_AVAILABLE = True
    print("Hardware monitoring available (Jetson device detected)")
except ImportError:
    print("Hardware monitoring not available (not a Jetson device)")


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def dataset_loaders(dataset_name='MNIST', test_batch_size=512):
    """Load dataset and return test loader with dataset properties."""
    if dataset_name == 'MNIST':
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x))])
        test_ds = MNIST('./data/', train=False, download=True, transform=transform)
        onehot_max_value = 10.0
        is_color = False
    elif dataset_name == 'FMNIST':
        transform = Compose([
            ToTensor(),
            Normalize((0.2860,), (0.3530,)),
            Lambda(lambda x: torch.flatten(x))])
        test_ds = FashionMNIST('./data/', train=False, download=True, transform=transform)
        onehot_max_value = 10.0
        is_color = False
    elif dataset_name == 'SVHN':
        transform = Compose([
            ToTensor(),
            Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            Lambda(lambda x: torch.flatten(x))])
        test_ds = SVHN('./data/', split='test', download=True, transform=transform)
        onehot_max_value = 10.0
        is_color = True
    elif dataset_name == 'CIFAR10':
        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            Lambda(lambda x: torch.flatten(x))])
        test_ds = CIFAR10('./data/', train=False, download=True, transform=transform)
        onehot_max_value = 10.0
        is_color = True
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    return test_loader, onehot_max_value, is_color


def load_model_to_device(model_path, device, onehot_max_value, is_color):
    """
    Load model fresh to specified device.
    Returns a new model instance on the target device.
    
    Note: We don't call model.eval() because the Net class overrides train()
    with a custom method for Forward-Forward training. The FF network doesn't
    use dropout or batch normalization, so eval mode isn't needed.
    """
    model = torch.load(model_path, map_location=device, weights_only=False)
    
    # Set model attributes
    model.device = device
    for layer in model.layers:
        layer.to(device)
    for softmax_layer in model.softmax_layers:
        softmax_layer.to(device)
    model.onehot_max_value = onehot_max_value
    model.is_color = is_color
    # Note: Don't call model.eval() - Net.train() is overridden for FF training
    
    return model


def get_width_from_model(model):
    """Extract hidden layer width from model architecture."""
    if hasattr(model, 'layers') and len(model.layers) > 0:
        return model.layers[0].out_features
    return 0


def measure_inference_latency(model, inputs, batch_size, device):
    """
    Measure inference latency for a forward pass.
    Returns latency in milliseconds and number of samples processed.
    """
    num_samples = len(inputs)
    
    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        _ = model.predict_one_pass(inputs, batch_size=batch_size)
    
    # Synchronize after inference
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000
    return latency_ms, num_samples


def measure_inference_with_energy(model, inputs, batch_size, device, power_monitor, hw_monitor=None):
    """
    Measure inference latency AND energy consumption for all three power rails.
    Works for both CPU and GPU inference.
    
    Args:
        model: The model to evaluate
        inputs: Input tensor
        batch_size: Batch size for inference
        device: torch device
        power_monitor: INA3221PowerMonitor instance
        hw_monitor: TegratsMonitor instance (optional, for background sampling)
    
    Returns:
        dict with latency_ms, num_samples, and energy/power for each rail:
        - VDD_IN: Total system power
        - VDD_CPU_GPU_CV: CPU/GPU/CV power  
        - VDD_SOC: SoC power
    """
    num_samples = len(inputs)
    
    # Power rail names
    rails = ['VDD_IN', 'VDD_CPU_GPU_CV', 'VDD_SOC']
    
    # Start inference measurement if using TegratsMonitor
    if hw_monitor:
        hw_monitor.start_inference_measurement()
    
    # Get power reading before
    power_before = power_monitor.get_power_metrics() if power_monitor else {}
    
    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        _ = model.predict_one_pass(inputs, batch_size=batch_size)
    
    # Synchronize after inference
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # Get power reading after
    power_after = power_monitor.get_power_metrics() if power_monitor else {}
    
    # Stop inference measurement
    energy_metrics = None
    if hw_monitor:
        energy_metrics = hw_monitor.stop_inference_measurement(num_samples)
    
    latency_ms = (end_time - start_time) * 1000
    latency_s = latency_ms / 1000.0
    
    result = {
        'latency_ms': latency_ms,
        'num_samples': num_samples,
        'latency_per_sample_ms': latency_ms / num_samples if num_samples > 0 else 0
    }
    
    # Calculate energy for each power rail
    total_energy_mj = 0
    for rail in rails:
        power_key = f'{rail}_power_mw'
        if power_before and power_after and power_key in power_before and power_key in power_after:
            # Use average of before/after power readings
            avg_power_mw = (power_before[power_key] + power_after[power_key]) / 2
            energy_mj = avg_power_mw * latency_s  # mW * s = mJ
            
            result[f'{rail}_power_mw'] = avg_power_mw
            result[f'{rail}_energy_mj'] = energy_mj
            result[f'{rail}_energy_per_sample_mj'] = energy_mj / num_samples if num_samples > 0 else 0
            
            if rail == 'VDD_IN':
                total_energy_mj = energy_mj
        else:
            result[f'{rail}_power_mw'] = 0
            result[f'{rail}_energy_mj'] = 0
            result[f'{rail}_energy_per_sample_mj'] = 0
    
    # Keep backward compatible fields using VDD_IN as the main energy metric
    result['energy_mj'] = total_energy_mj
    result['avg_power_mw'] = result.get('VDD_IN_power_mw', 0)
    result['energy_per_sample_mj'] = total_energy_mj / num_samples if num_samples > 0 else 0
    
    return result


def measure_per_sample_latencies(model, inputs, device, max_samples=100):
    """
    Measure latency for individual samples (batch_size=1).
    Returns list of per-sample latencies in milliseconds.
    
    Args:
        model: The model to evaluate
        inputs: Test inputs tensor
        device: torch device
        max_samples: Maximum number of samples to measure (for speed)
    """
    per_sample_latencies = []
    num_samples = min(len(inputs), max_samples)
    
    # Warmup run (discarded)
    with torch.no_grad():
        _ = model.predict_one_pass(inputs[:1], batch_size=1)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    for i in range(num_samples):
        sample = inputs[i:i+1]
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model.predict_one_pass(sample, batch_size=1)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        per_sample_latencies.append(latency_ms)
    
    return per_sample_latencies


def measure_per_sample_energy(model, inputs, device, power_monitor, max_samples=50):
    """
    Measure energy for individual samples (batch_size=1) across all power rails.
    Returns dict with per-sample energy for each rail.
    
    Note: Energy measurement at batch_size=1 has high variance due to
    power sensor sampling rate. Use larger sample counts for better estimates.
    """
    rails = ['VDD_IN', 'VDD_CPU_GPU_CV', 'VDD_SOC']
    per_sample_data = {rail: [] for rail in rails}
    per_sample_data['powers'] = {rail: [] for rail in rails}
    
    num_samples = min(len(inputs), max_samples)
    
    if not power_monitor:
        return per_sample_data
    
    # Warmup run (discarded)
    with torch.no_grad():
        _ = model.predict_one_pass(inputs[:1], batch_size=1)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    for i in range(num_samples):
        sample = inputs[i:i+1]
        
        # Get power before
        power_before = power_monitor.get_power_metrics()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model.predict_one_pass(sample, batch_size=1)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Get power after
        power_after = power_monitor.get_power_metrics()
        
        latency_s = end_time - start_time
        
        # Calculate energy for each rail
        for rail in rails:
            power_key = f'{rail}_power_mw'
            if power_key in power_before and power_key in power_after:
                avg_power_mw = (power_before[power_key] + power_after[power_key]) / 2
                energy_mj = avg_power_mw * latency_s
                per_sample_data[rail].append(energy_mj)
                per_sample_data['powers'][rail].append(avg_power_mw)
    
    return per_sample_data


def create_violin_plot_comparison(data_cuda, data_cpu, output_path):
    """
    Create side-by-side violin plots comparing CUDA vs CPU latency by width.
    """
    widths = sorted(set(data_cuda.keys()) | set(data_cpu.keys()))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # CUDA plot
    if data_cuda:
        cuda_widths = sorted(data_cuda.keys())
        cuda_data = [data_cuda[w] for w in cuda_widths]
        parts = axes[0].violinplot(cuda_data, positions=range(len(cuda_widths)), showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#2ecc71')
            pc.set_alpha(0.7)
        axes[0].set_xticks(range(len(cuda_widths)))
        axes[0].set_xticklabels([str(w) for w in cuda_widths])
        axes[0].set_xlabel('Network Width')
        axes[0].set_ylabel('Latency per Sample (ms)')
        axes[0].set_title('CUDA (GPU)')
        axes[0].yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Add stats
        for i, width in enumerate(cuda_widths):
            mean_val = np.mean(data_cuda[width])
            std_val = np.std(data_cuda[width])
            axes[0].annotate(f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                            xy=(i, max(data_cuda[width]) * 1.02),
                            ha='center', va='bottom', fontsize=7)
    
    # CPU plot
    if data_cpu:
        cpu_widths = sorted(data_cpu.keys())
        cpu_data = [data_cpu[w] for w in cpu_widths]
        parts = axes[1].violinplot(cpu_data, positions=range(len(cpu_widths)), showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#e74c3c')
            pc.set_alpha(0.7)
        axes[1].set_xticks(range(len(cpu_widths)))
        axes[1].set_xticklabels([str(w) for w in cpu_widths])
        axes[1].set_xlabel('Network Width')
        axes[1].set_title('CPU')
        axes[1].yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Add stats
        for i, width in enumerate(cpu_widths):
            mean_val = np.mean(data_cpu[width])
            std_val = np.std(data_cpu[width])
            axes[1].annotate(f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                            xy=(i, max(data_cpu[width]) * 1.02),
                            ha='center', va='bottom', fontsize=7)
    
    plt.suptitle('Inference Latency Distribution: CUDA vs CPU', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved violin comparison plot to: {output_path}")


def create_throughput_plot(throughput_data, output_path):
    """
    Create throughput vs batch size plot for each width, comparing CUDA and CPU.
    
    throughput_data: dict[(width, batch_size, device)] -> list of throughput values
    """
    # Organize data
    plot_data = defaultdict(lambda: {'cuda': {}, 'cpu': {}})
    
    for (width, batch_size, device), throughputs in throughput_data.items():
        mean_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)
        plot_data[width][device][batch_size] = (mean_throughput, std_throughput)
    
    widths = sorted(plot_data.keys())
    n_widths = len(widths)
    
    # Create subplot for each width
    cols = min(3, n_widths)
    rows = (n_widths + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    
    for idx, width in enumerate(widths):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        # CUDA data
        if plot_data[width]['cuda']:
            batch_sizes = sorted(plot_data[width]['cuda'].keys())
            means = [plot_data[width]['cuda'][bs][0] for bs in batch_sizes]
            stds = [plot_data[width]['cuda'][bs][1] for bs in batch_sizes]
            ax.errorbar(batch_sizes, means, yerr=stds, marker='o', label='CUDA', 
                       color='#2ecc71', capsize=3, linewidth=2)
        
        # CPU data
        if plot_data[width]['cpu']:
            batch_sizes = sorted(plot_data[width]['cpu'].keys())
            means = [plot_data[width]['cpu'][bs][0] for bs in batch_sizes]
            stds = [plot_data[width]['cpu'][bs][1] for bs in batch_sizes]
            ax.errorbar(batch_sizes, means, yerr=stds, marker='s', label='CPU', 
                       color='#e74c3c', capsize=3, linewidth=2)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title(f'Width: {width}')
        ax.set_xscale('log', base=2)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide empty subplots
    for idx in range(n_widths, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Throughput vs Batch Size by Network Width', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved throughput plot to: {output_path}")


def create_speedup_heatmap(latency_data, output_path):
    """
    Create heatmap showing GPU speedup ratio (CPU_time / GPU_time) by width and batch size.
    """
    # Organize data
    speedup_data = {}
    
    # Get all widths and batch sizes
    widths = set()
    batch_sizes = set()
    for (width, batch_size, device) in latency_data.keys():
        widths.add(width)
        batch_sizes.add(batch_size)
    
    widths = sorted(widths)
    batch_sizes = sorted(batch_sizes)
    
    # Calculate speedup for each (width, batch_size)
    speedup_matrix = np.full((len(widths), len(batch_sizes)), np.nan)
    
    for i, width in enumerate(widths):
        for j, batch_size in enumerate(batch_sizes):
            cuda_key = (width, batch_size, 'cuda')
            cpu_key = (width, batch_size, 'cpu')
            
            if cuda_key in latency_data and cpu_key in latency_data:
                cuda_mean = np.mean(latency_data[cuda_key])
                cpu_mean = np.mean(latency_data[cpu_key])
                if cuda_mean > 0:
                    speedup_matrix[i, j] = cpu_mean / cuda_mean
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speedup (CPU time / GPU time)')
    
    # Set ticks
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.set_yticks(range(len(widths)))
    ax.set_yticklabels([str(w) for w in widths])
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Network Width')
    ax.set_title('GPU Speedup over CPU (higher = GPU faster)')
    
    # Add text annotations
    for i in range(len(widths)):
        for j in range(len(batch_sizes)):
            value = speedup_matrix[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > 2 or value < 0.5 else 'black'
                ax.text(j, i, f'{value:.1f}x', ha='center', va='center', 
                       color=text_color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved speedup heatmap to: {output_path}")


def create_per_sample_histogram(per_sample_data, output_path):
    """
    Create overlaid histogram of per-sample latencies for GPU vs CPU.
    
    per_sample_data: dict[(width, device)] -> list of per-sample latencies
    """
    widths = sorted(set(w for w, d in per_sample_data.keys()))
    
    n_widths = len(widths)
    cols = min(3, n_widths)
    rows = (n_widths + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    
    for idx, width in enumerate(widths):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        cuda_key = (width, 'cuda')
        cpu_key = (width, 'cpu')
        
        if cuda_key in per_sample_data:
            ax.hist(per_sample_data[cuda_key], bins=30, alpha=0.7, 
                   label='CUDA', color='#2ecc71', density=True)
        
        if cpu_key in per_sample_data:
            ax.hist(per_sample_data[cpu_key], bins=30, alpha=0.7, 
                   label='CPU', color='#e74c3c', density=True)
        
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Density')
        ax.set_title(f'Width: {width}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide empty subplots
    for idx in range(n_widths, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Per-Sample Latency Distribution (batch_size=1)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved per-sample histogram to: {output_path}")


def create_single_violin_plot(data_by_width, output_path, ylabel='Energy per Sample (mJ)', title='Energy Distribution by Width', color='#3498db'):
    """
    Create a single violin plot for one metric by width.
    """
    if not data_by_width:
        print(f"No data for violin plot: {output_path}")
        return
    
    widths = sorted(data_by_width.keys())
    data = [data_by_width[w] for w in widths]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    parts = ax.violinplot(data, positions=range(len(widths)), showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel('Network Width')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add stats
    for i, width in enumerate(widths):
        mean_val = np.mean(data_by_width[width])
        std_val = np.std(data_by_width[width])
        ax.annotate(f'μ={mean_val:.4f}\nσ={std_val:.4f}', 
                    xy=(i, max(data_by_width[width]) * 1.02),
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved violin plot to: {output_path}")


def create_energy_plot(energy_data, output_path, rail='VDD_IN'):
    """
    Create energy per sample vs batch size plot for each width, comparing CUDA and CPU.
    
    energy_data: dict[(width, batch_size, device, rail)] -> list of energy_per_sample values
    """
    # Organize data for both devices
    plot_data = defaultdict(lambda: {'cuda': {}, 'cpu': {}})
    
    for key, energies in energy_data.items():
        if len(key) == 4:
            width, batch_size, device, data_rail = key
            if data_rail == rail:
                mean_energy = np.mean(energies)
                std_energy = np.std(energies)
                plot_data[width][device][batch_size] = (mean_energy, std_energy)
        elif len(key) == 3:
            # Backward compatibility: (width, batch_size, device)
            width, batch_size, device = key
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)
            plot_data[width][device][batch_size] = (mean_energy, std_energy)
    
    widths = sorted(plot_data.keys())
    n_widths = len(widths)
    
    if n_widths == 0:
        print(f"No energy data for plot ({rail})")
        return
    
    # Create subplot for each width
    cols = min(3, n_widths)
    rows = (n_widths + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    
    for idx, width in enumerate(widths):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        # CUDA data
        if plot_data[width]['cuda']:
            batch_sizes = sorted(plot_data[width]['cuda'].keys())
            means = [plot_data[width]['cuda'][bs][0] for bs in batch_sizes]
            stds = [plot_data[width]['cuda'][bs][1] for bs in batch_sizes]
            ax.errorbar(batch_sizes, means, yerr=stds, marker='o', label='CUDA',
                       color='#2ecc71', capsize=3, linewidth=2)
        
        # CPU data
        if plot_data[width]['cpu']:
            batch_sizes = sorted(plot_data[width]['cpu'].keys())
            means = [plot_data[width]['cpu'][bs][0] for bs in batch_sizes]
            stds = [plot_data[width]['cpu'][bs][1] for bs in batch_sizes]
            ax.errorbar(batch_sizes, means, yerr=stds, marker='s', label='CPU',
                       color='#e74c3c', capsize=3, linewidth=2)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Energy per Sample (mJ)')
        ax.set_title(f'Width: {width}')
        ax.set_xscale('log', base=2)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide empty subplots
    for idx in range(n_widths, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f'Energy per Sample vs Batch Size ({rail})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved energy plot to: {output_path}")


def create_energy_comparison_plot(energy_data_by_rail, output_path):
    """
    Create a comparison plot showing all three power rails for both CUDA and CPU.
    
    energy_data_by_rail: dict[rail] -> dict[(width, batch_size, device)] -> list of values
    """
    rails = ['VDD_IN', 'VDD_CPU_GPU_CV', 'VDD_SOC']
    rail_colors = {'VDD_IN': '#3498db', 'VDD_CPU_GPU_CV': '#e74c3c', 'VDD_SOC': '#2ecc71'}
    
    # Get all widths
    all_widths = set()
    for rail_data in energy_data_by_rail.values():
        for key in rail_data.keys():
            all_widths.add(key[0])
    widths = sorted(all_widths)
    
    if not widths:
        print("No data for energy comparison plot")
        return
    
    # Create figure with subplots for CUDA and CPU
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax_idx, device in enumerate(['cuda', 'cpu']):
        ax = axes[ax_idx]
        
        for rail in rails:
            if rail not in energy_data_by_rail:
                continue
            
            rail_data = energy_data_by_rail[rail]
            
            # Get mean energy per width at batch_size=32
            width_energies = []
            valid_widths = []
            for width in widths:
                key = (width, 32, device)
                if key in rail_data and rail_data[key]:
                    width_energies.append(np.mean(rail_data[key]))
                    valid_widths.append(width)
            
            if valid_widths:
                ax.plot(range(len(valid_widths)), width_energies, 'o-', 
                       label=rail, color=rail_colors[rail], linewidth=2, markersize=8)
        
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_xlabel('Network Width')
        ax.set_ylabel('Energy per Sample (mJ)')
        ax.set_title(f'{device.upper()} - Power Rails Comparison (batch_size=32)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('Energy Consumption by Power Rail', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved energy comparison plot to: {output_path}")


def create_energy_heatmap(energy_data, output_path, device='cuda', rail='VDD_IN'):
    """
    Create heatmap showing energy per sample by width and batch size for a specific device and rail.
    """
    # Get all widths and batch sizes for specified device
    widths = set()
    batch_sizes = set()
    for key in energy_data.keys():
        if len(key) >= 3:
            w, bs, dev = key[0], key[1], key[2]
            if dev == device:
                widths.add(w)
                batch_sizes.add(bs)
    
    widths = sorted(widths)
    batch_sizes = sorted(batch_sizes)
    
    if not widths or not batch_sizes:
        print(f"No data for energy heatmap ({device}, {rail})")
        return
    
    # Build energy matrix
    energy_matrix = np.full((len(widths), len(batch_sizes)), np.nan)
    
    for i, width in enumerate(widths):
        for j, batch_size in enumerate(batch_sizes):
            key = (width, batch_size, device)
            if key in energy_data and energy_data[key]:
                energy_matrix[i, j] = np.mean(energy_data[key])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(energy_matrix, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Energy per Sample (mJ)')
    
    # Set ticks
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.set_yticks(range(len(widths)))
    ax.set_yticklabels([str(w) for w in widths])
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Network Width')
    ax.set_title(f'Energy per Sample ({device.upper()}, {rail})')
    
    # Add text annotations
    for i in range(len(widths)):
        for j in range(len(batch_sizes)):
            value = energy_matrix[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                       color='black', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved energy heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate inference variance with GPU/CPU comparison')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--sweep-id', type=str, default='9xqabsi3', help='W&B sweep ID')
    parser.add_argument('--project', type=str, default='edgeff-network-width', help='W&B project name')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name')
    parser.add_argument('--iterations', type=int, default=50, help='Number of inference iterations per config')
    parser.add_argument('--batch-sizes', type=str, default='1,8,32,128,512', help='Comma-separated batch sizes')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples to use')
    parser.add_argument('--per-sample-count', type=int, default=100, help='Samples for per-sample latency measurement')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for plots')
    parser.add_argument('--skip-cpu', action='store_true', help='Skip CPU evaluation (faster)')
    parser.add_argument('--skip-cuda', action='store_true', help='Skip CUDA evaluation')
    parser.add_argument('--skip-energy', action='store_true', help='Skip energy measurement')
    parser.add_argument('--hw-interval-ms', type=int, default=100, help='Hardware monitoring interval in ms')
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        sweep_id = config.get('sweep_id', args.sweep_id)
        project_name = config.get('project', args.project)
        dataset_name = config.get('dataset', args.dataset)
        iterations = config.get('iterations', args.iterations)
        batch_sizes = config.get('batch_sizes', [1, 8, 32, 128, 512])
        num_samples = config.get('num_samples', args.num_samples)
        per_sample_count = config.get('per_sample_count', args.per_sample_count)
        output_dir = config.get('output_dir', args.output_dir)
        skip_cpu = config.get('skip_cpu', args.skip_cpu)
        skip_cuda = config.get('skip_cuda', args.skip_cuda)
        skip_energy = config.get('skip_energy', args.skip_energy)
        hw_interval_ms = config.get('hw_interval_ms', args.hw_interval_ms)
    else:
        sweep_id = args.sweep_id
        project_name = args.project
        dataset_name = args.dataset
        iterations = args.iterations
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
        num_samples = args.num_samples
        per_sample_count = args.per_sample_count
        output_dir = args.output_dir
        skip_cpu = args.skip_cpu
        skip_cuda = args.skip_cuda
        skip_energy = args.skip_energy
        hw_interval_ms = args.hw_interval_ms

    # Determine devices to test
    devices = []
    if not skip_cuda and torch.cuda.is_available():
        devices.append('cuda')
    if not skip_cpu:
        devices.append('cpu')
    
    if not devices:
        print("Error: No devices to test. Check --skip-cpu and --skip-cuda flags.")
        return

    print(f"Testing on devices: {devices}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Iterations per config: {iterations}")

    # Initialize hardware monitoring (Jetson only)
    power_monitor = None
    hw_monitor = None
    energy_available = False
    
    if HW_MONITOR_AVAILABLE and not skip_energy:
        power_monitor = INA3221PowerMonitor()
        if power_monitor.hwmon_path:
            energy_available = True
            print(f"Energy measurement enabled via INA3221")
            # Start background monitoring for better power sampling
            hw_monitor = TegratsMonitor(power_monitor=power_monitor, interval_ms=hw_interval_ms)
            hw_monitor.start()
        else:
            print("INA3221 not found - energy measurement disabled")
            power_monitor = None
    else:
        print("Energy measurement disabled or not available")

    # Load .env
    root_dir = Path(__file__).resolve().parent.parent.parent
    dotenv_path = root_dir / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

    # Initialize wandb
    wandb.init(project=project_name, job_type='variance-eval', config={
        'sweep_id': sweep_id,
        'iterations': iterations,
        'batch_sizes': batch_sizes,
        'num_samples': num_samples,
        'dataset': dataset_name,
        'devices': devices,
        'energy_measurement': energy_available
    })

    # Get runs from sweep
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    runs = list(sweep.runs)
    print(f"Found {len(runs)} runs in sweep {sweep_id}")

    # Load dataset (to CPU first, will move to device later)
    test_loader, onehot_max_value, is_color = dataset_loaders(dataset_name, test_batch_size=512)
    test_inputs_cpu = torch.cat([d for d, _ in test_loader], dim=0)
    test_targets_cpu = torch.cat([t for _, t in test_loader], dim=0)

    # Limit samples if specified
    if num_samples < len(test_inputs_cpu):
        test_inputs_cpu = test_inputs_cpu[:num_samples]
        test_targets_cpu = test_targets_cpu[:num_samples]
    
    print(f"Using {len(test_inputs_cpu)} test samples")

    # Data storage
    latency_data = defaultdict(list)  # (width, batch_size, device) -> latency_per_sample values
    throughput_data = defaultdict(list)  # (width, batch_size, device) -> throughput values
    
    # Energy data organized by power rail
    rails = ['VDD_IN', 'VDD_CPU_GPU_CV', 'VDD_SOC']
    energy_data = {rail: defaultdict(list) for rail in rails}  # rail -> (width, batch_size, device) -> values
    power_data = {rail: defaultdict(list) for rail in rails}  # rail -> (width, batch_size, device) -> values
    
    per_sample_latency_data = defaultdict(list)  # (width, device) -> per-sample latencies
    per_sample_energy_data = {rail: defaultdict(list) for rail in rails}  # rail -> (width, device) -> values
    
    # For violin plots by width (default batch size = 32)
    latency_by_width_cuda = defaultdict(list)
    latency_by_width_cpu = defaultdict(list)
    energy_by_width = {rail: defaultdict(list) for rail in rails}  # rail -> width -> values
    
    all_measurements = []
    
    # Process each run
    for run_idx, run in enumerate(runs):
        print(f"\n{'='*60}")
        print(f"Processing run {run_idx + 1}/{len(runs)}: {run.id}")
        
        # Get model artifact
        artifacts = list(run.logged_artifacts())
        model_artifact = None
        for art in artifacts:
            if art.type == 'model':
                model_artifact = art
                break
        
        if model_artifact is None:
            print(f"  No model artifact found, skipping")
            continue

        # Download model
        artifact_dir = model_artifact.download()
        model_path = os.path.join(artifact_dir, 'temp_')
        
        if not os.path.exists(model_path):
            model_files = [f for f in os.listdir(artifact_dir) 
                          if f.endswith('.pt') or f.endswith('.pth') or 'temp' in f]
            if model_files:
                model_path = os.path.join(artifact_dir, model_files[0])
            else:
                print(f"  Model file not found in {artifact_dir}, skipping")
                continue

        # Get width from a temporary model load
        temp_model = torch.load(model_path, map_location='cpu', weights_only=False)
        width = get_width_from_model(temp_model)
        if model_artifact.metadata and 'width' in model_artifact.metadata:
            width = model_artifact.metadata['width']
        del temp_model
        
        print(f"  Width: {width}")

        # Test on each device
        for device_name in devices:
            device = torch.device(device_name)
            print(f"\n  Device: {device_name.upper()}")
            
            # Load model fresh for this device
            model = load_model_to_device(model_path, device, onehot_max_value, is_color)
            
            # Move test data to device
            test_inputs = test_inputs_cpu.to(device)
            
            # Test each batch size
            for batch_size in batch_sizes:
                print(f"    Batch size: {batch_size}", end=" ")
                
                try:
                    # Warmup run (discarded) - fresh for each batch size
                    with torch.no_grad():
                        _ = model.predict_one_pass(test_inputs[:batch_size], batch_size=batch_size)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    # Timed iterations
                    batch_latencies = []
                    batch_throughputs = []
                    batch_energies = {rail: [] for rail in rails}
                    batch_powers = {rail: [] for rail in rails}
                    
                    for i in range(iterations):
                        if energy_available:
                            # Measure with energy (for both CPU and GPU)
                            metrics = measure_inference_with_energy(
                                model, test_inputs, batch_size, device, 
                                power_monitor, hw_monitor
                            )
                            latency_ms = metrics['latency_ms']
                            n_samples = metrics['num_samples']
                            latency_per_sample = metrics['latency_per_sample_ms']
                            
                            # Store energy for each rail
                            for rail in rails:
                                energy_key = f'{rail}_energy_per_sample_mj'
                                power_key = f'{rail}_power_mw'
                                if energy_key in metrics:
                                    batch_energies[rail].append(metrics[energy_key])
                                    batch_powers[rail].append(metrics[power_key])
                                    energy_data[rail][(width, batch_size, device_name)].append(metrics[energy_key])
                                    power_data[rail][(width, batch_size, device_name)].append(metrics[power_key])
                        else:
                            # Measure latency only
                            latency_ms, n_samples = measure_inference_latency(
                                model, test_inputs, batch_size, device
                            )
                            latency_per_sample = latency_ms / n_samples
                        
                        throughput = n_samples / (latency_ms / 1000)  # samples/sec
                        
                        batch_latencies.append(latency_per_sample)
                        batch_throughputs.append(throughput)
                        
                        # Store measurement
                        key = (width, batch_size, device_name)
                        latency_data[key].append(latency_per_sample)
                        throughput_data[key].append(throughput)
                        
                        measurement = {
                            'run_id': run.id,
                            'width': width,
                            'batch_size': batch_size,
                            'device': device_name,
                            'iteration': i,
                            'latency_per_sample_ms': latency_per_sample,
                            'throughput_samples_sec': throughput,
                            'total_latency_ms': latency_ms,
                            'num_samples': n_samples
                        }
                        
                        if energy_available:
                            for rail in rails:
                                energy_key = f'{rail}_energy_per_sample_mj'
                                power_key = f'{rail}_power_mw'
                                if energy_key in metrics:
                                    measurement[f'{rail}_energy_mj'] = metrics[energy_key]
                                    measurement[f'{rail}_power_mw'] = metrics[power_key]
                        
                        all_measurements.append(measurement)
                        
                        # Log to wandb
                        log_dict = {
                            'width': width,
                            'batch_size': batch_size,
                            'device': device_name,
                            'latency_per_sample_ms': latency_per_sample,
                            'throughput_samples_sec': throughput
                        }
                        if energy_available:
                            for rail in rails:
                                energy_key = f'{rail}_energy_per_sample_mj'
                                power_key = f'{rail}_power_mw'
                                if energy_key in metrics:
                                    log_dict[f'{rail}_energy_mj'] = metrics[energy_key]
                                    log_dict[f'{rail}_power_mw'] = metrics[power_key]
                        wandb.log(log_dict)
                    
                    mean_lat = np.mean(batch_latencies)
                    std_lat = np.std(batch_latencies)
                    mean_thr = np.mean(batch_throughputs)
                    
                    result_str = f"| Latency: {mean_lat:.4f}±{std_lat:.4f} ms | Throughput: {mean_thr:.1f} samples/sec"
                    
                    if batch_energies['VDD_IN']:
                        mean_energy = np.mean(batch_energies['VDD_IN'])
                        std_energy = np.std(batch_energies['VDD_IN'])
                        result_str += f" | VDD_IN: {mean_energy:.4f}±{std_energy:.4f} mJ"
                        # Also show CPU/GPU rail
                        if batch_energies['VDD_CPU_GPU_CV']:
                            mean_cpugpu = np.mean(batch_energies['VDD_CPU_GPU_CV'])
                            result_str += f" | CPU_GPU: {mean_cpugpu:.4f} mJ"
                    
                    print(result_str)
                    
                    # Store for violin plots (use batch_size=32 as default)
                    if batch_size == 32:
                        if device_name == 'cuda':
                            latency_by_width_cuda[width].extend(batch_latencies)
                        else:
                            latency_by_width_cpu[width].extend(batch_latencies)
                        
                        # Store energy for all rails
                        for rail in rails:
                            if batch_energies[rail]:
                                energy_by_width[rail][width].extend(batch_energies[rail])
                            
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"| OOM - skipping")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    else:
                        print(f"| Error: {e}")
                    continue
            
            # Per-sample latency measurement (batch_size=1)
            print(f"    Measuring per-sample latencies ({per_sample_count} samples)...", end=" ")
            try:
                per_sample_latencies = measure_per_sample_latencies(
                    model, test_inputs, device, max_samples=per_sample_count
                )
                per_sample_latency_data[(width, device_name)].extend(per_sample_latencies)
                print(f"Mean: {np.mean(per_sample_latencies):.4f} ms")
            except Exception as e:
                print(f"Error: {e}")
            
            # Per-sample energy measurement (for both CPU and GPU, if available)
            if energy_available:
                print(f"    Measuring per-sample energy ({min(per_sample_count, 50)} samples)...", end=" ")
                try:
                    per_sample_result = measure_per_sample_energy(
                        model, test_inputs, device, power_monitor, max_samples=min(per_sample_count, 50)
                    )
                    if per_sample_result['VDD_IN']:
                        for rail in rails:
                            if per_sample_result[rail]:
                                per_sample_energy_data[rail][(width, device_name)].extend(per_sample_result[rail])
                        mean_vdd_in = np.mean(per_sample_result['VDD_IN'])
                        print(f"Mean VDD_IN: {mean_vdd_in:.4f} mJ")
                    else:
                        print("No data")
                except Exception as e:
                    print(f"Error: {e}")
            
            # Clean up
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Stop hardware monitoring
    if hw_monitor:
        hw_monitor.stop()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save raw measurements to CSV
    df = pd.DataFrame(all_measurements)
    csv_path = os.path.join(output_dir, 'latency_measurements.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw measurements to: {csv_path}")

    # Compute and save statistics
    stats_data = []
    for (width, batch_size, device_name), latencies in sorted(latency_data.items()):
        throughputs = throughput_data[(width, batch_size, device_name)]
        stats_entry = {
            'width': width,
            'batch_size': batch_size,
            'device': device_name,
            'latency_mean_ms': np.mean(latencies),
            'latency_std_ms': np.std(latencies),
            'latency_min_ms': np.min(latencies),
            'latency_max_ms': np.max(latencies),
            'latency_p5_ms': np.percentile(latencies, 5),
            'latency_p95_ms': np.percentile(latencies, 95),
            'throughput_mean': np.mean(throughputs),
            'throughput_std': np.std(throughputs),
            'n_measurements': len(latencies)
        }
        
        # Add energy stats for each rail if available
        energy_key = (width, batch_size, device_name)
        for rail in rails:
            if energy_key in energy_data[rail] and energy_data[rail][energy_key]:
                energies = energy_data[rail][energy_key]
                stats_entry[f'{rail}_energy_mean_mj'] = np.mean(energies)
                stats_entry[f'{rail}_energy_std_mj'] = np.std(energies)
                stats_entry[f'{rail}_energy_min_mj'] = np.min(energies)
                stats_entry[f'{rail}_energy_max_mj'] = np.max(energies)
                
                powers = power_data[rail].get(energy_key, [])
                if powers:
                    stats_entry[f'{rail}_power_mean_mw'] = np.mean(powers)
                    stats_entry[f'{rail}_power_std_mw'] = np.std(powers)
        
        stats_data.append(stats_entry)
    
    stats_df = pd.DataFrame(stats_data)
    stats_path = os.path.join(output_dir, 'latency_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to: {stats_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)
    print(stats_df.to_string(index=False))

    # Create plots
    print("\nGenerating plots...")
    
    # 1. Violin plot comparison (latency)
    violin_path = os.path.join(output_dir, 'violin_latency_by_width.png')
    create_violin_plot_comparison(latency_by_width_cuda, latency_by_width_cpu, violin_path)
    
    # 2. Violin plots for energy by rail
    for rail in rails:
        if energy_by_width[rail]:
            violin_energy_path = os.path.join(output_dir, f'violin_energy_{rail}_by_width.png')
            create_single_violin_plot(energy_by_width[rail], violin_energy_path, 
                                      ylabel='Energy per Sample (mJ)',
                                      title=f'{rail} Energy Distribution by Width (batch_size=32)',
                                      color='#9b59b6' if rail == 'VDD_IN' else '#e74c3c' if rail == 'VDD_CPU_GPU_CV' else '#2ecc71')
    
    # 3. Throughput vs batch size
    throughput_path = os.path.join(output_dir, 'throughput_vs_batchsize.png')
    create_throughput_plot(throughput_data, throughput_path)
    
    # 4. Energy vs batch size for each rail (comparing CPU and GPU)
    for rail in rails:
        if energy_data[rail]:
            energy_plot_path = os.path.join(output_dir, f'energy_{rail}_vs_batchsize.png')
            create_energy_plot(energy_data[rail], energy_plot_path, rail=rail)
    
    # 5. Energy comparison plot (all three rails side by side)
    if any(energy_data[rail] for rail in rails):
        comparison_path = os.path.join(output_dir, 'energy_rails_comparison.png')
        create_energy_comparison_plot(energy_data, comparison_path)
    
    # 6. Speedup heatmap
    speedup_path = os.path.join(output_dir, 'speedup_heatmap.png')
    create_speedup_heatmap(latency_data, speedup_path)
    
    # 7. Energy heatmaps for each rail and device
    for rail in rails:
        if energy_data[rail]:
            for device in ['cuda', 'cpu']:
                energy_hm_path = os.path.join(output_dir, f'energy_heatmap_{rail}_{device}.png')
                create_energy_heatmap(energy_data[rail], energy_hm_path, device=device, rail=rail)
    
    # 8. Per-sample histogram
    if per_sample_latency_data:
        histogram_path = os.path.join(output_dir, 'per_sample_histogram.png')
        create_per_sample_histogram(per_sample_latency_data, histogram_path)

    # Log plots to wandb
    plots_to_log = {
        'violin_latency_plot': violin_path,
        'throughput_plot': throughput_path,
        'speedup_heatmap': speedup_path,
    }
    
    # Add energy plots by rail
    for rail in rails:
        if energy_by_width[rail]:
            plots_to_log[f'violin_energy_{rail}'] = os.path.join(output_dir, f'violin_energy_{rail}_by_width.png')
        if energy_data[rail]:
            plots_to_log[f'energy_{rail}_plot'] = os.path.join(output_dir, f'energy_{rail}_vs_batchsize.png')
    
    if any(energy_data[rail] for rail in rails):
        plots_to_log['energy_rails_comparison'] = os.path.join(output_dir, 'energy_rails_comparison.png')
    
    if per_sample_latency_data:
        plots_to_log['per_sample_latency_histogram'] = os.path.join(output_dir, 'per_sample_histogram.png')
    
    for name, path in plots_to_log.items():
        if os.path.exists(path):
            wandb.log({name: wandb.Image(path)})
    
    wandb.log({'latency_stats_table': wandb.Table(dataframe=stats_df)})

    wandb.finish()
    print("\nDone!")


if __name__ == '__main__':
    main()
