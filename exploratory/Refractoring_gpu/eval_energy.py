"""
eval_energy.py - Bimodal Energy Distribution Analysis
======================================================

This script investigates bimodal energy distributions observed during neural
network inference on Jetson devices. It runs controlled inference activity
patterns to expose and characterize the dual-mode energy behavior.

Hypothesis: Bimodality may be caused by GPU DVFS state transitions (idle→active),
power sensor aliasing, or cache/memory effects.

Features:
    - Multiple inference activity patterns (burst, continuous, cold-start)
    - Variable cooldown periods between inferences
    - Bimodality detection via Gaussian Mixture Models (GMM)
    - Hartigan's dip test for statistical bimodality confirmation
    - Mode labeling and per-mode statistics
    - Time-series analysis of sequential measurements

Usage:
    python eval_energy.py --config eval_energy_config.yaml
    python eval_energy.py --model-path ./model.pt --iterations 200

Output:
    - bimodal_histogram.png: Histogram with KDE and GMM component overlay
    - energy_timeseries.png: Sequential measurements with mode coloring
    - latency_vs_energy_scatter.png: Scatter plot colored by mode
    - mode_violin_comparison.png: Split violin comparing low/high modes
    - cooldown_effect.png: Energy distribution by cooldown duration
    - bimodality_stats.csv: GMM parameters, dip test results
    - energy_measurements.csv: Raw data with mode labels
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
from matplotlib.patches import Patch
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from scipy import stats
from scipy.stats import gaussian_kde

# Optional: GMM for bimodality analysis
try:
    from sklearn.mixture import GaussianMixture
    GMM_AVAILABLE = True
except ImportError:
    GMM_AVAILABLE = False
    print("sklearn not available - GMM fitting disabled")

# Optional: Dip test for bimodality
try:
    import diptest
    DIPTEST_AVAILABLE = True
except ImportError:
    DIPTEST_AVAILABLE = False
    print("diptest not available - statistical bimodality test disabled")


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
    """Load YAML configuration file."""
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
    
    return model


def get_width_from_model(model):
    """Extract hidden layer width from model architecture."""
    if hasattr(model, 'layers') and len(model.layers) > 0:
        return model.layers[0].out_features
    return 0


def measure_inference_with_energy(model, inputs, batch_size, device, power_monitor, hw_monitor=None):
    """
    Measure inference latency AND energy consumption for all three power rails.
    
    Returns:
        dict with latency_ms, num_samples, and energy/power for each rail
    """
    num_samples = len(inputs)
    rails = ['VDD_IN', 'VDD_CPU_GPU_CV', 'VDD_SOC']
    
    if hw_monitor:
        hw_monitor.start_inference_measurement()
    
    power_before = power_monitor.get_power_metrics() if power_monitor else {}
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        _ = model.predict_one_pass(inputs, batch_size=batch_size)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    power_after = power_monitor.get_power_metrics() if power_monitor else {}
    
    if hw_monitor:
        hw_monitor.stop_inference_measurement(num_samples)
    
    latency_ms = (end_time - start_time) * 1000
    latency_s = latency_ms / 1000.0
    
    result = {
        'latency_ms': latency_ms,
        'num_samples': num_samples,
        'latency_per_sample_ms': latency_ms / num_samples if num_samples > 0 else 0,
        'timestamp': time.time()
    }
    
    total_energy_mj = 0
    for rail in rails:
        power_key = f'{rail}_power_mw'
        if power_before and power_after and power_key in power_before and power_key in power_after:
            avg_power_mw = (power_before[power_key] + power_after[power_key]) / 2
            energy_mj = avg_power_mw * latency_s
            
            result[f'{rail}_power_mw'] = avg_power_mw
            result[f'{rail}_energy_mj'] = energy_mj
            result[f'{rail}_energy_per_sample_mj'] = energy_mj / num_samples if num_samples > 0 else 0
            
            if rail == 'VDD_IN':
                total_energy_mj = energy_mj
        else:
            result[f'{rail}_power_mw'] = 0
            result[f'{rail}_energy_mj'] = 0
            result[f'{rail}_energy_per_sample_mj'] = 0
    
    result['energy_mj'] = total_energy_mj
    result['energy_per_sample_mj'] = total_energy_mj / num_samples if num_samples > 0 else 0
    
    return result


# =============================================================================
# INFERENCE ACTIVITY PATTERNS
# =============================================================================

def run_continuous_inference(model, inputs, batch_size, device, power_monitor, 
                             hw_monitor, iterations):
    """
    Run back-to-back inference with no delays between iterations.
    This pattern should keep GPU in active state, potentially showing
    only the "low energy" mode.
    """
    measurements = []
    
    # Warmup
    with torch.no_grad():
        _ = model.predict_one_pass(inputs[:batch_size], batch_size=batch_size)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    for i in range(iterations):
        metrics = measure_inference_with_energy(
            model, inputs, batch_size, device, power_monitor, hw_monitor
        )
        metrics['pattern'] = 'continuous'
        metrics['iteration'] = i
        metrics['cooldown_ms'] = 0
        measurements.append(metrics)
    
    return measurements


def run_burst_with_cooldown(model, inputs, batch_size, device, power_monitor,
                            hw_monitor, iterations, cooldown_ms):
    """
    Run inference with fixed cooldown between iterations.
    Varying cooldown should reveal transition between energy modes.
    """
    measurements = []
    cooldown_s = cooldown_ms / 1000.0
    
    # Warmup
    with torch.no_grad():
        _ = model.predict_one_pass(inputs[:batch_size], batch_size=batch_size)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    for i in range(iterations):
        # Cooldown before inference
        if i > 0:
            time.sleep(cooldown_s)
        
        metrics = measure_inference_with_energy(
            model, inputs, batch_size, device, power_monitor, hw_monitor
        )
        metrics['pattern'] = f'burst_{cooldown_ms}ms'
        metrics['iteration'] = i
        metrics['cooldown_ms'] = cooldown_ms
        measurements.append(metrics)
    
    return measurements


def run_cold_start_sequence(model, inputs, batch_size, device, power_monitor,
                            hw_monitor, iterations, cold_delay_s=2.0):
    """
    Run inference after extended idle periods to capture "cold start" behavior.
    Each inference is preceded by a long delay to let GPU return to idle.
    """
    measurements = []
    
    for i in range(iterations):
        # Long delay to ensure GPU returns to idle state
        time.sleep(cold_delay_s)
        
        # No warmup - we want to capture cold start
        metrics = measure_inference_with_energy(
            model, inputs, batch_size, device, power_monitor, hw_monitor
        )
        metrics['pattern'] = 'cold_start'
        metrics['iteration'] = i
        metrics['cooldown_ms'] = cold_delay_s * 1000
        measurements.append(metrics)
    
    return measurements


def run_alternating_pattern(model, inputs, batch_size, device, power_monitor,
                            hw_monitor, iterations, burst_count=5, gap_ms=500):
    """
    Alternating bursts of continuous inference followed by gaps.
    This creates a mixed pattern that should show both modes.
    """
    measurements = []
    gap_s = gap_ms / 1000.0
    
    # Warmup
    with torch.no_grad():
        _ = model.predict_one_pass(inputs[:batch_size], batch_size=batch_size)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    iteration = 0
    while iteration < iterations:
        # Burst phase
        for b in range(min(burst_count, iterations - iteration)):
            metrics = measure_inference_with_energy(
                model, inputs, batch_size, device, power_monitor, hw_monitor
            )
            metrics['pattern'] = 'alternating_burst'
            metrics['iteration'] = iteration
            metrics['cooldown_ms'] = 0
            metrics['burst_position'] = b
            measurements.append(metrics)
            iteration += 1
        
        # Gap phase
        if iteration < iterations:
            time.sleep(gap_s)
            
            metrics = measure_inference_with_energy(
                model, inputs, batch_size, device, power_monitor, hw_monitor
            )
            metrics['pattern'] = 'alternating_gap'
            metrics['iteration'] = iteration
            metrics['cooldown_ms'] = gap_ms
            metrics['burst_position'] = -1  # After gap
            measurements.append(metrics)
            iteration += 1
    
    return measurements


# =============================================================================
# BIMODALITY ANALYSIS
# =============================================================================

def fit_gmm(data, n_components=2):
    """
    Fit a Gaussian Mixture Model to detect bimodality.
    
    Returns:
        dict with GMM parameters, component assignments, and metrics
    """
    if not GMM_AVAILABLE:
        return None
    
    data = np.array(data).reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)
    
    # Get component assignments (0 = low mode, 1 = high mode typically)
    labels = gmm.predict(data)
    probabilities = gmm.predict_proba(data)
    
    # Sort components by mean (so component 0 is always the lower one)
    means = gmm.means_.flatten()
    sort_idx = np.argsort(means)
    
    sorted_means = means[sort_idx]
    sorted_stds = np.sqrt(gmm.covariances_.flatten())[sort_idx]
    sorted_weights = gmm.weights_[sort_idx]
    
    # Remap labels to sorted order
    label_map = {old: new for new, old in enumerate(sort_idx)}
    sorted_labels = np.array([label_map[l] for l in labels])
    
    # Mode separation metric (distance between means in std units)
    pooled_std = np.sqrt(np.sum(sorted_weights * sorted_stds**2))
    mode_separation = (sorted_means[1] - sorted_means[0]) / pooled_std if pooled_std > 0 else 0
    
    return {
        'means': sorted_means,
        'stds': sorted_stds,
        'weights': sorted_weights,
        'labels': sorted_labels,
        'probabilities': probabilities,
        'mode_separation': mode_separation,
        'bic': gmm.bic(data),
        'aic': gmm.aic(data),
        'gmm': gmm
    }


def run_dip_test(data):
    """
    Run Hartigan's dip test for unimodality.
    
    Returns:
        dict with dip statistic and p-value
        Lower p-value = more likely bimodal
    """
    if not DIPTEST_AVAILABLE:
        return None
    
    data = np.array(data)
    dip_stat, p_value = diptest.diptest(data)
    
    return {
        'dip_statistic': dip_stat,
        'p_value': p_value,
        'is_bimodal': p_value < 0.05  # Conventional threshold
    }


def compute_mode_statistics(data, labels):
    """
    Compute per-mode statistics.
    """
    data = np.array(data)
    labels = np.array(labels)
    
    stats_by_mode = {}
    for mode in np.unique(labels):
        mode_data = data[labels == mode]
        stats_by_mode[f'mode_{mode}'] = {
            'count': len(mode_data),
            'mean': np.mean(mode_data),
            'std': np.std(mode_data),
            'min': np.min(mode_data),
            'max': np.max(mode_data),
            'median': np.median(mode_data),
            'p5': np.percentile(mode_data, 5),
            'p95': np.percentile(mode_data, 95)
        }
    
    return stats_by_mode


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_bimodal_histogram(data, gmm_result, output_path, title='Energy Distribution',
                             xlabel='Energy per Sample (mJ)'):
    """
    Create histogram with KDE and GMM component overlay.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = np.array(data)
    
    # Histogram
    n, bins, patches = ax.hist(data, bins=50, density=True, alpha=0.6, 
                                color='steelblue', edgecolor='black', label='Data')
    
    # KDE
    try:
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), 'k-', linewidth=2, label='KDE')
    except Exception:
        pass
    
    # GMM components
    if gmm_result and 'gmm' in gmm_result:
        x_range = np.linspace(data.min(), data.max(), 200)
        colors = ['#e74c3c', '#2ecc71']  # Red for low, green for high
        
        for i, (mean, std, weight) in enumerate(zip(
            gmm_result['means'], gmm_result['stds'], gmm_result['weights']
        )):
            # Gaussian component
            component = weight * stats.norm.pdf(x_range, mean, std)
            ax.plot(x_range, component, '--', color=colors[i], linewidth=2,
                   label=f'Mode {i}: μ={mean:.4f}, σ={std:.4f}, w={weight:.2f}')
            
            # Vertical line at mean
            ax.axvline(mean, color=colors[i], linestyle=':', alpha=0.7)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add bimodality info
    if gmm_result:
        info_text = f"Mode Separation: {gmm_result['mode_separation']:.2f}σ"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved bimodal histogram to: {output_path}")


def create_energy_timeseries(measurements, gmm_result, output_path):
    """
    Create time-series plot of sequential energy measurements with mode coloring.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    energies = [m['energy_per_sample_mj'] for m in measurements]
    latencies = [m['latency_per_sample_ms'] for m in measurements]
    iterations = list(range(len(measurements)))
    
    colors = ['#e74c3c', '#2ecc71']  # Red for low mode, green for high mode
    
    # Energy time series
    if gmm_result and 'labels' in gmm_result:
        labels = gmm_result['labels']
        for i, (it, e, lab) in enumerate(zip(iterations, energies, labels)):
            axes[0].scatter(it, e, c=colors[lab], s=30, alpha=0.7)
    else:
        axes[0].plot(iterations, energies, 'o-', markersize=4, alpha=0.7)
    
    axes[0].set_ylabel('Energy per Sample (mJ)')
    axes[0].set_title('Energy Time Series with Mode Assignment')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Add pattern annotations
    prev_pattern = None
    for i, m in enumerate(measurements):
        pattern = m.get('pattern', 'unknown')
        if pattern != prev_pattern:
            axes[0].axvline(i, color='gray', linestyle='--', alpha=0.3)
            axes[0].text(i, axes[0].get_ylim()[1], pattern, rotation=90, 
                        fontsize=8, va='top')
            prev_pattern = pattern
    
    # Latency time series (for correlation)
    if gmm_result and 'labels' in gmm_result:
        for i, (it, lat, lab) in enumerate(zip(iterations, latencies, labels)):
            axes[1].scatter(it, lat, c=colors[lab], s=30, alpha=0.7)
    else:
        axes[1].plot(iterations, latencies, 'o-', markersize=4, alpha=0.7)
    
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Latency per Sample (ms)')
    axes[1].set_title('Latency Time Series')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Legend
    if gmm_result:
        legend_elements = [
            Patch(facecolor=colors[0], label='Low Energy Mode'),
            Patch(facecolor=colors[1], label='High Energy Mode')
        ]
        axes[0].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved energy timeseries to: {output_path}")


def create_latency_energy_scatter(measurements, gmm_result, output_path):
    """
    Scatter plot of latency vs energy, colored by mode assignment.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    energies = np.array([m['energy_per_sample_mj'] for m in measurements])
    latencies = np.array([m['latency_per_sample_ms'] for m in measurements])
    
    colors_map = ['#e74c3c', '#2ecc71']
    
    if gmm_result and 'labels' in gmm_result:
        labels = gmm_result['labels']
        colors = [colors_map[l] for l in labels]
        scatter = ax.scatter(latencies, energies, c=colors, s=50, alpha=0.7, edgecolors='black')
        
        # Add legend
        legend_elements = [
            Patch(facecolor=colors_map[0], label=f'Low Mode (n={np.sum(labels==0)})'),
            Patch(facecolor=colors_map[1], label=f'High Mode (n={np.sum(labels==1)})')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        # Add mode centroids
        for mode in [0, 1]:
            mask = labels == mode
            if np.any(mask):
                ax.scatter(np.mean(latencies[mask]), np.mean(energies[mask]),
                          marker='X', s=200, c=colors_map[mode], edgecolors='black',
                          linewidths=2, zorder=10)
    else:
        ax.scatter(latencies, energies, s=50, alpha=0.7, edgecolors='black')
    
    ax.set_xlabel('Latency per Sample (ms)')
    ax.set_ylabel('Energy per Sample (mJ)')
    ax.set_title('Latency vs Energy (colored by mode)')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add correlation
    corr = np.corrcoef(latencies, energies)[0, 1]
    ax.text(0.98, 0.02, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved latency-energy scatter to: {output_path}")


def create_mode_violin_comparison(measurements, gmm_result, output_path):
    """
    Split violin plot comparing distributions of low vs high energy modes.
    """
    if not gmm_result or 'labels' not in gmm_result:
        print("No GMM result for violin comparison")
        return
    
    labels = gmm_result['labels']
    
    # Collect data by mode
    metrics_to_plot = ['energy_per_sample_mj', 'latency_per_sample_ms', 'VDD_CPU_GPU_CV_power_mw']
    metric_labels = ['Energy (mJ)', 'Latency (ms)', 'CPU/GPU Power (mW)']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 6))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    colors = ['#e74c3c', '#2ecc71']
    
    for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
        data_by_mode = [[], []]
        for m, lab in zip(measurements, labels):
            if metric in m:
                data_by_mode[lab].append(m[metric])
        
        # Filter out empty modes
        valid_data = []
        valid_labels = []
        valid_positions = []
        for i, d in enumerate(data_by_mode):
            if d:
                valid_data.append(d)
                valid_labels.append(f'Mode {i}')
                valid_positions.append(i)
        
        if valid_data:
            parts = ax.violinplot(valid_data, positions=valid_positions, 
                                  showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[valid_positions[i]])
                pc.set_alpha(0.7)
            
            ax.set_xticks(valid_positions)
            ax.set_xticklabels(valid_labels)
        
        ax.set_ylabel(label)
        ax.set_title(f'{label} by Mode')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add stats
        for i, d in enumerate(valid_data):
            mean_val = np.mean(d)
            std_val = np.std(d)
            ax.annotate(f'μ={mean_val:.4f}\nσ={std_val:.4f}',
                       xy=(valid_positions[i], max(d) * 1.02),
                       ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Metric Distributions by Energy Mode', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved mode violin comparison to: {output_path}")


def create_cooldown_effect_plot(measurements, output_path):
    """
    Show how energy distribution changes with cooldown duration.
    """
    # Group measurements by cooldown
    by_cooldown = defaultdict(list)
    for m in measurements:
        cooldown = m.get('cooldown_ms', 0)
        by_cooldown[cooldown].append(m['energy_per_sample_mj'])
    
    cooldowns = sorted(by_cooldown.keys())
    
    if len(cooldowns) < 2:
        print("Not enough cooldown variations for plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    data = [by_cooldown[c] for c in cooldowns]
    bp = axes[0].boxplot(data, labels=[f'{c}ms' for c in cooldowns], patch_artist=True)
    
    # Color by cooldown (gradient)
    cmap = plt.cm.coolwarm
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cmap(i / len(cooldowns)))
    
    axes[0].set_xlabel('Cooldown Duration')
    axes[0].set_ylabel('Energy per Sample (mJ)')
    axes[0].set_title('Energy Distribution by Cooldown Duration')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Mean energy vs cooldown
    means = [np.mean(by_cooldown[c]) for c in cooldowns]
    stds = [np.std(by_cooldown[c]) for c in cooldowns]
    
    axes[1].errorbar(cooldowns, means, yerr=stds, marker='o', capsize=5,
                     linewidth=2, markersize=8, color='steelblue')
    axes[1].set_xlabel('Cooldown Duration (ms)')
    axes[1].set_ylabel('Mean Energy per Sample (mJ)')
    axes[1].set_title('Mean Energy vs Cooldown Duration')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Add trend indicator
    if len(cooldowns) > 2:
        corr = np.corrcoef(cooldowns, means)[0, 1]
        axes[1].text(0.98, 0.02, f'Correlation: {corr:.3f}', transform=axes[1].transAxes,
                    ha='right', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved cooldown effect plot to: {output_path}")


def create_pattern_comparison_plot(measurements, output_path):
    """
    Compare energy distributions across different inference patterns.
    """
    # Group by pattern
    by_pattern = defaultdict(list)
    for m in measurements:
        pattern = m.get('pattern', 'unknown')
        by_pattern[pattern].append(m['energy_per_sample_mj'])
    
    patterns = sorted(by_pattern.keys())
    
    if len(patterns) < 2:
        print("Not enough patterns for comparison plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = [by_pattern[p] for p in patterns]
    positions = range(len(patterns))
    
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    
    # Color coding
    colors = plt.cm.Set2(np.linspace(0, 1, len(patterns)))
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(patterns, rotation=45, ha='right')
    ax.set_ylabel('Energy per Sample (mJ)')
    ax.set_title('Energy Distribution by Inference Pattern')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add stats
    for i, (pattern, d) in enumerate(zip(patterns, data)):
        mean_val = np.mean(d)
        std_val = np.std(d)
        ax.annotate(f'n={len(d)}\nμ={mean_val:.4f}',
                   xy=(i, max(d) * 1.02),
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved pattern comparison plot to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze bimodal energy distributions')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model file')
    parser.add_argument('--sweep-id', type=str, default=None, help='W&B sweep ID (uses first model)')
    parser.add_argument('--project', type=str, default='edgeff-network-width', help='W&B project name')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--iterations', type=int, default=200, help='Total iterations per pattern')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples to use')
    parser.add_argument('--cooldowns', type=str, default='0,50,100,200,500,1000', 
                        help='Comma-separated cooldown durations (ms)')
    parser.add_argument('--output-dir', type=str, default='./energy_analysis', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--hw-interval-ms', type=int, default=100, help='Hardware monitoring interval')
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        model_path = config.get('model_path', args.model_path)
        sweep_id = config.get('sweep_id', args.sweep_id)
        project_name = config.get('project', args.project)
        dataset_name = config.get('dataset', args.dataset)
        batch_size = config.get('batch_size', args.batch_size)
        iterations = config.get('iterations', args.iterations)
        num_samples = config.get('num_samples', args.num_samples)
        cooldowns = config.get('cooldowns', [0, 50, 100, 200, 500, 1000])
        output_dir = config.get('output_dir', args.output_dir)
        device_name = config.get('device', args.device)
        hw_interval_ms = config.get('hw_interval_ms', args.hw_interval_ms)
    else:
        model_path = args.model_path
        sweep_id = args.sweep_id
        project_name = args.project
        dataset_name = args.dataset
        batch_size = args.batch_size
        iterations = args.iterations
        num_samples = args.num_samples
        cooldowns = [int(x) for x in args.cooldowns.split(',')]
        output_dir = args.output_dir
        device_name = args.device
        hw_interval_ms = args.hw_interval_ms

    # Validate device
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    device = torch.device(device_name)
    
    print(f"Device: {device_name}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations per pattern: {iterations}")
    print(f"Cooldown durations: {cooldowns} ms")

    # Check energy measurement availability
    if not HW_MONITOR_AVAILABLE:
        print("ERROR: Hardware monitoring not available. This script requires Jetson device.")
        return

    # Initialize hardware monitoring
    power_monitor = INA3221PowerMonitor()
    if not power_monitor.hwmon_path:
        print("ERROR: INA3221 power sensor not found.")
        return
    
    print("Energy measurement enabled via INA3221")
    hw_monitor = TegratsMonitor(power_monitor=power_monitor, interval_ms=hw_interval_ms)
    hw_monitor.start()

    # Load .env
    root_dir = Path(__file__).resolve().parent.parent.parent
    dotenv_path = root_dir / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

    # Get model path
    if model_path is None and sweep_id:
        # Get first model from sweep
        api = wandb.Api()
        sweep = api.sweep(f"{project_name}/{sweep_id}")
        runs = list(sweep.runs)
        
        for run in runs:
            artifacts = list(run.logged_artifacts())
            for art in artifacts:
                if art.type == 'model':
                    artifact_dir = art.download()
                    model_path = os.path.join(artifact_dir, 'temp_')
                    if not os.path.exists(model_path):
                        model_files = [f for f in os.listdir(artifact_dir) 
                                      if f.endswith('.pt') or f.endswith('.pth') or 'temp' in f]
                        if model_files:
                            model_path = os.path.join(artifact_dir, model_files[0])
                    break
            if model_path:
                break
    
    if not model_path or not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        hw_monitor.stop()
        return

    print(f"Using model: {model_path}")

    # Initialize wandb
    wandb.init(project=project_name, job_type='energy-bimodal-analysis', config={
        'model_path': model_path,
        'batch_size': batch_size,
        'iterations': iterations,
        'cooldowns': cooldowns,
        'device': device_name,
        'dataset': dataset_name
    })

    # Load dataset
    test_loader, onehot_max_value, is_color = dataset_loaders(dataset_name, test_batch_size=512)
    test_inputs_cpu = torch.cat([d for d, _ in test_loader], dim=0)
    
    if num_samples < len(test_inputs_cpu):
        test_inputs_cpu = test_inputs_cpu[:num_samples]
    
    print(f"Using {len(test_inputs_cpu)} test samples")

    # Load model
    model = load_model_to_device(model_path, device, onehot_max_value, is_color)
    width = get_width_from_model(model)
    print(f"Model width: {width}")

    # Move test data to device
    test_inputs = test_inputs_cpu.to(device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================================================
    # RUN INFERENCE PATTERNS
    # ==========================================================================
    
    all_measurements = []
    
    # 1. Continuous inference (no delays)
    print("\n" + "="*60)
    print("Pattern 1: Continuous Inference")
    print("="*60)
    continuous_measurements = run_continuous_inference(
        model, test_inputs, batch_size, device, power_monitor, hw_monitor, iterations
    )
    all_measurements.extend(continuous_measurements)
    print(f"  Collected {len(continuous_measurements)} measurements")
    
    # 2. Burst with varying cooldowns
    for cooldown in cooldowns:
        if cooldown == 0:
            continue  # Already covered by continuous
        
        print(f"\nPattern 2: Burst with {cooldown}ms cooldown")
        print("-"*40)
        burst_measurements = run_burst_with_cooldown(
            model, test_inputs, batch_size, device, power_monitor, hw_monitor,
            iterations // len(cooldowns), cooldown
        )
        all_measurements.extend(burst_measurements)
        print(f"  Collected {len(burst_measurements)} measurements")
    
    # 3. Cold start sequence
    print("\n" + "="*60)
    print("Pattern 3: Cold Start Sequence")
    print("="*60)
    cold_measurements = run_cold_start_sequence(
        model, test_inputs, batch_size, device, power_monitor, hw_monitor,
        iterations // 4, cold_delay_s=2.0
    )
    all_measurements.extend(cold_measurements)
    print(f"  Collected {len(cold_measurements)} measurements")
    
    # 4. Alternating pattern
    print("\n" + "="*60)
    print("Pattern 4: Alternating Bursts")
    print("="*60)
    alternating_measurements = run_alternating_pattern(
        model, test_inputs, batch_size, device, power_monitor, hw_monitor,
        iterations // 2, burst_count=5, gap_ms=500
    )
    all_measurements.extend(alternating_measurements)
    print(f"  Collected {len(alternating_measurements)} measurements")

    # Stop hardware monitoring
    hw_monitor.stop()

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================
    
    print("\n" + "="*60)
    print("BIMODALITY ANALYSIS")
    print("="*60)
    
    # Extract energy values
    energies = [m['energy_per_sample_mj'] for m in all_measurements]
    
    # Fit GMM
    gmm_result = fit_gmm(energies, n_components=2)
    if gmm_result:
        print(f"\nGMM Results:")
        print(f"  Mode 0 (Low):  mean={gmm_result['means'][0]:.4f}, std={gmm_result['stds'][0]:.4f}, weight={gmm_result['weights'][0]:.2f}")
        print(f"  Mode 1 (High): mean={gmm_result['means'][1]:.4f}, std={gmm_result['stds'][1]:.4f}, weight={gmm_result['weights'][1]:.2f}")
        print(f"  Mode Separation: {gmm_result['mode_separation']:.2f}σ")
        print(f"  BIC: {gmm_result['bic']:.2f}, AIC: {gmm_result['aic']:.2f}")
        
        # Add mode labels to measurements
        for m, label in zip(all_measurements, gmm_result['labels']):
            m['mode_label'] = int(label)
    
    # Run dip test
    dip_result = run_dip_test(energies)
    if dip_result:
        print(f"\nDip Test Results:")
        print(f"  Dip Statistic: {dip_result['dip_statistic']:.4f}")
        print(f"  P-value: {dip_result['p_value']:.4f}")
        print(f"  Is Bimodal (p<0.05): {dip_result['is_bimodal']}")
    
    # Mode statistics
    if gmm_result:
        mode_stats = compute_mode_statistics(energies, gmm_result['labels'])
        print(f"\nPer-Mode Statistics:")
        for mode, stats in mode_stats.items():
            print(f"  {mode}: n={stats['count']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # ==========================================================================
    # SAVE DATA
    # ==========================================================================
    
    # Save raw measurements
    df = pd.DataFrame(all_measurements)
    csv_path = os.path.join(output_dir, 'energy_measurements.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved measurements to: {csv_path}")
    
    # Save bimodality statistics
    bimodal_stats = {
        'n_measurements': len(all_measurements),
        'energy_mean': np.mean(energies),
        'energy_std': np.std(energies),
        'energy_min': np.min(energies),
        'energy_max': np.max(energies)
    }
    
    if gmm_result:
        bimodal_stats.update({
            'gmm_mode0_mean': gmm_result['means'][0],
            'gmm_mode0_std': gmm_result['stds'][0],
            'gmm_mode0_weight': gmm_result['weights'][0],
            'gmm_mode1_mean': gmm_result['means'][1],
            'gmm_mode1_std': gmm_result['stds'][1],
            'gmm_mode1_weight': gmm_result['weights'][1],
            'gmm_mode_separation': gmm_result['mode_separation'],
            'gmm_bic': gmm_result['bic'],
            'gmm_aic': gmm_result['aic']
        })
    
    if dip_result:
        bimodal_stats.update({
            'dip_statistic': dip_result['dip_statistic'],
            'dip_p_value': dip_result['p_value'],
            'is_bimodal': dip_result['is_bimodal']
        })
    
    stats_df = pd.DataFrame([bimodal_stats])
    stats_path = os.path.join(output_dir, 'bimodality_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved bimodality stats to: {stats_path}")

    # ==========================================================================
    # CREATE PLOTS
    # ==========================================================================
    
    print("\nGenerating plots...")
    
    # 1. Bimodal histogram with GMM overlay
    histogram_path = os.path.join(output_dir, 'bimodal_histogram.png')
    create_bimodal_histogram(energies, gmm_result, histogram_path,
                            title=f'Energy Distribution (width={width}, batch_size={batch_size})')
    
    # 2. Energy time series
    timeseries_path = os.path.join(output_dir, 'energy_timeseries.png')
    create_energy_timeseries(all_measurements, gmm_result, timeseries_path)
    
    # 3. Latency vs energy scatter
    scatter_path = os.path.join(output_dir, 'latency_vs_energy_scatter.png')
    create_latency_energy_scatter(all_measurements, gmm_result, scatter_path)
    
    # 4. Mode violin comparison
    violin_path = os.path.join(output_dir, 'mode_violin_comparison.png')
    create_mode_violin_comparison(all_measurements, gmm_result, violin_path)
    
    # 5. Cooldown effect plot
    cooldown_path = os.path.join(output_dir, 'cooldown_effect.png')
    create_cooldown_effect_plot(all_measurements, cooldown_path)
    
    # 6. Pattern comparison
    pattern_path = os.path.join(output_dir, 'pattern_comparison.png')
    create_pattern_comparison_plot(all_measurements, pattern_path)

    # Log to wandb
    plots_to_log = {
        'bimodal_histogram': histogram_path,
        'energy_timeseries': timeseries_path,
        'latency_energy_scatter': scatter_path,
        'mode_violin_comparison': violin_path,
        'cooldown_effect': cooldown_path,
        'pattern_comparison': pattern_path
    }
    
    for name, path in plots_to_log.items():
        if os.path.exists(path):
            wandb.log({name: wandb.Image(path)})
    
    wandb.log(bimodal_stats)
    wandb.log({'measurements_table': wandb.Table(dataframe=df)})
    
    wandb.finish()
    
    # Cleanup
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
