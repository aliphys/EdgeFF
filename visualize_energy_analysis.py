import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_json_files(log_dir: str = "logs") -> Dict[int, List[Dict]]:
    """Load all width JSON files (all 5 runs)."""
    widths = [100, 200, 300]
    data = {}
    
    for width in widths:
        runs = []
        for run in range(1, 6):  # Load runs 1-5
            json_path = Path(log_dir) / f"width{width}run{run}.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    runs.append(json.load(f))
                print(f"✓ Loaded width{width}run{run}.json")
            else:
                print(f"⚠ Missing: {json_path}")
        
        if runs:
            data[width] = runs
    
    return data

def aggregate_runs(runs: List[Dict]) -> Dict:
    """Aggregate statistics across multiple runs."""
    if not runs:
        return {}
    
    # Use first run as template
    aggregated = {
        "model_info": runs[0]["model_info"],
        "configuration": runs[0]["configuration"],
        "layer_analysis": {},
        "specialization": {},
        "energy_analysis": {},
        "dataset_results": {}
    }
    
    # Aggregate dataset results (test accuracy, etc.)
    if "dataset_results" in runs[0]:
        aggregated["dataset_results"] = {}
        for dataset_type in ["test", "validation", "train"]:
            if dataset_type in runs[0]["dataset_results"]:
                dataset_dict = {}
                for metric in ["accuracy", "f1_score", "error"]:
                    if metric in runs[0]["dataset_results"][dataset_type]:
                        values = [r["dataset_results"][dataset_type][metric] for r in runs 
                                 if dataset_type in r.get("dataset_results", {})]
                        dataset_dict[f"{metric}_mean"] = np.mean(values)
                        dataset_dict[f"{metric}_std"] = np.std(values)
                        dataset_dict[f"{metric}_values"] = values
                aggregated["dataset_results"][dataset_type] = dataset_dict
    
    # Aggregate layer analysis
    if "layer_analysis" in runs[0] and "fixed_layer_accuracy" in runs[0]["layer_analysis"]:
        layer_acc = {}
        for layer_key in runs[0]["layer_analysis"]["fixed_layer_accuracy"].keys():
            values = [r["layer_analysis"]["fixed_layer_accuracy"][layer_key] for r in runs]
            layer_acc[layer_key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values
            }
        aggregated["layer_analysis"]["fixed_layer_accuracy"] = layer_acc
    
    # Aggregate specialization scores
    if "specialization" in runs[0] and "scores" in runs[0]["specialization"]:
        spec_scores = {}
        for layer_key in runs[0]["specialization"]["scores"].keys():
            values = [r["specialization"]["scores"][layer_key] for r in runs]
            spec_scores[layer_key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values
            }
        aggregated["specialization"]["scores"] = spec_scores
    
    # Aggregate baseline energy
    # Check if communication power is included
    has_comm_power = "per_sample_mj_with_comm" in runs[0]["energy_analysis"]["baseline"]
    # Check if latency is included
    has_latency = "average_latency_ms" in runs[0]["energy_analysis"]["baseline"]
    
    if has_comm_power:
        baseline_values = [r["energy_analysis"]["baseline"]["per_sample_mj_with_comm"] for r in runs]
    else:
        baseline_values = [r["energy_analysis"]["baseline"]["per_sample_mj"] for r in runs]
    
    aggregated["energy_analysis"]["baseline"] = {
        "per_sample_mj_mean": np.mean(baseline_values),
        "per_sample_mj_std": np.std(baseline_values),
        "per_sample_mj_values": baseline_values,
        "total_samples": runs[0]["energy_analysis"]["baseline"]["total_samples"],
        "has_comm_power": has_comm_power,
        "has_latency": has_latency
    }
    
    # Add baseline latency if available
    if has_latency:
        baseline_latency_values = [r["energy_analysis"]["baseline"]["average_latency_ms"] for r in runs]
        aggregated["energy_analysis"]["baseline"]["average_latency_ms_mean"] = np.mean(baseline_latency_values)
        aggregated["energy_analysis"]["baseline"]["average_latency_ms_std"] = np.std(baseline_latency_values)
    
    # Aggregate multiplier results
    multipliers = [r["multiplier"] for r in runs[0]["energy_analysis"]["multiplier_results"]]
    aggregated["energy_analysis"]["multiplier_results"] = []
    
    for mult in multipliers:
        mult_data = {
            "multiplier": mult,
            "per_sample_mj_mean": 0,
            "per_sample_mj_std": 0,
            "energy_savings_pct_mean": 0,
            "energy_savings_pct_std": 0,
            "accuracy_mean": 0,
            "accuracy_std": 0,
            "exit_distribution": []
        }
        
        # Collect values across runs
        per_sample_values = []
        savings_values = []
        accuracy_values = []
        latency_values = []
        
        for run in runs:
            for result in run["energy_analysis"]["multiplier_results"]:
                if result["multiplier"] == mult:
                    # Use per_sample_mj_with_comm if available, otherwise per_sample_mj
                    if has_comm_power and "per_sample_mj_with_comm" in result:
                        per_sample_values.append(result["per_sample_mj_with_comm"])
                    else:
                        per_sample_values.append(result["per_sample_mj"])
                    
                    savings_values.append(result["energy_savings_pct"])
                    
                    # Get cumulative accuracy from last exit layer
                    if result["exit_distribution"]:
                        accuracy_values.append(result["exit_distribution"][-1]["cumulative_accuracy"] * 100)
                    
                    # Get latency if available
                    if has_latency and "average_latency_ms" in result:
                        latency_values.append(result["average_latency_ms"])
                    
                    break
        
        mult_data["per_sample_mj_mean"] = np.mean(per_sample_values)
        mult_data["per_sample_mj_std"] = np.std(per_sample_values)
        mult_data["energy_savings_pct_mean"] = np.mean(savings_values)
        mult_data["energy_savings_pct_std"] = np.std(savings_values)
        mult_data["accuracy_mean"] = np.mean(accuracy_values)
        mult_data["accuracy_std"] = np.std(accuracy_values)
        
        # Add latency if available
        if has_latency and latency_values:
            mult_data["average_latency_ms_mean"] = np.mean(latency_values)
            mult_data["average_latency_ms_std"] = np.std(latency_values)
        
        # Aggregate exit distribution (use mean across runs)
        exit_layers = [1, 2, 3]
        for layer in exit_layers:
            layer_samples = []
            layer_accuracies = []
            layer_energies = []
            layer_cpu_gpu = []
            layer_soc = []
            
            for run in runs:
                for result in run["energy_analysis"]["multiplier_results"]:
                    if result["multiplier"] == mult:
                        for exit_info in result["exit_distribution"]:
                            if exit_info["layer"] == layer:
                                layer_samples.append(exit_info["samples"])
                                layer_accuracies.append(exit_info["accuracy"])
                                layer_energies.append(exit_info["per_sample_mj"])
                                layer_cpu_gpu.append(exit_info["cpu_gpu_power_w"])
                                layer_soc.append(exit_info["soc_power_w"])
                                break
                        break
            
            if layer_samples:
                mult_data["exit_distribution"].append({
                    "layer": layer,
                    "samples_mean": np.mean(layer_samples),
                    "samples_std": np.std(layer_samples),
                    "accuracy_mean": np.mean(layer_accuracies),
                    "per_sample_mj_mean": np.mean(layer_energies),
                    "cpu_gpu_power_w_mean": np.mean(layer_cpu_gpu),
                    "soc_power_w_mean": np.mean(layer_soc),
                })
        
        aggregated["energy_analysis"]["multiplier_results"].append(mult_data)
    
    return aggregated

def plot_energy_accuracy_tradeoff(data: Dict[int, Dict], output_dir: str = "output"):
    """Plot 1: Energy vs Accuracy trade-off across network widths with error bars."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'100': '#2ecc71', '200': '#3498db', '300': '#e74c3c'}
    markers = {'100': 'o', '200': 's', '300': '^'}
    
    for width, width_data in sorted(data.items()):
        multipliers = []
        energies = []
        energy_errors = []
        accuracies = []
        accuracy_errors = []
        
        for result in width_data['energy_analysis']['multiplier_results']:
            multipliers.append(result['multiplier'])
            energies.append(result['per_sample_mj_mean'])
            energy_errors.append(result['per_sample_mj_std'])
            accuracies.append(result['accuracy_mean'])
            accuracy_errors.append(result['accuracy_std'])
        
        # Plot line with markers and error bars
        ax.errorbar(energies, accuracies, 
                    xerr=energy_errors,
                    yerr=accuracy_errors,
                    marker=markers[str(width)], 
                    color=colors[str(width)],
                    linewidth=2.5, 
                    markersize=10,
                    label=f'Width {width}',
                    alpha=0.8,
                    capsize=5,
                    capthick=2)
        
        # Annotate multipliers
        for i, mult in enumerate(multipliers):
            ax.annotate(f'{mult}×', 
                       (energies[i], accuracies[i]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center',
                       fontsize=8,
                       alpha=0.7)
    
    ax.set_xlabel('Energy per Sample (mJ)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Energy-Accuracy Trade-off with Early Exit\nAcross Network Widths (n=5 runs)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(False)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'energy_accuracy_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_energy_savings_comparison(data: Dict[int, Dict], output_dir: str = "output"):
    """Plot 2: Energy savings percentage across multipliers and widths with error bars."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    multipliers = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    width_labels = []
    savings_data = {mult: [] for mult in multipliers}
    savings_errors = {mult: [] for mult in multipliers}
    
    for width in sorted(data.keys()):
        width_labels.append(f'Width {width}')
        
        for result in data[width]['energy_analysis']['multiplier_results']:
            mult = result['multiplier']
            if mult in multipliers:
                savings_data[mult].append(result['energy_savings_pct_mean'])
                savings_errors[mult].append(result['energy_savings_pct_std'])
    
    x = np.arange(len(width_labels))
    width = 0.13
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(multipliers)))
    
    for i, mult in enumerate(multipliers):
        offset = width * (i - len(multipliers)/2 + 0.5)
        bars = ax.bar(x + offset, savings_data[mult], width, 
                     yerr=savings_errors[mult],
                     label=f'{mult}×', color=colors[i], alpha=0.8,
                     capsize=3)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Network Width', fontsize=14, fontweight='bold')
    ax.set_ylabel('Energy Savings (%)', fontsize=14, fontweight='bold')
    ax.set_title('Energy Savings by Confidence Multiplier\nAcross Network Widths (n=5 runs)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(width_labels)
    ax.legend(title='Multiplier', fontsize=10, loc='upper left', ncol=2)
    ax.grid(False)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'energy_savings_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_exit_distribution(data: Dict[int, Dict], output_dir: str = "output"):
    """Plot 3: Exit layer distribution across widths with error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    multipliers_to_plot = [1.0, 2.0, 3.0]
    colors = ['#3498db', '#e74c3c', '#f39c12']
    
    for idx, (width, width_data) in enumerate(sorted(data.items())):
        ax = axes[idx]
        
        for result in width_data['energy_analysis']['multiplier_results']:
            if result['multiplier'] in multipliers_to_plot:
                layers = []
                percentages = []
                errors = []
                
                for exit_info in result['exit_distribution']:
                    layer = exit_info['layer']
                    samples_mean = exit_info['samples_mean']
                    samples_std = exit_info['samples_std']
                    total_samples = width_data['energy_analysis']['baseline']['total_samples']
                    percentage = (samples_mean / total_samples) * 100
                    error = (samples_std / total_samples) * 100
                    
                    layers.append(f"Layer {layer}")
                    percentages.append(percentage)
                    errors.append(error)
                
                mult_idx = multipliers_to_plot.index(result['multiplier'])
                x_offset = mult_idx * 0.25
                
                ax.bar([i + x_offset for i in range(len(layers))], 
                      percentages,
                      yerr=errors,
                      width=0.23,
                      label=f'{result["multiplier"]}×',
                      color=colors[mult_idx],
                      alpha=0.8,
                      capsize=3)
        
        ax.set_xlabel('Exit Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('% of Samples', fontsize=12, fontweight='bold')
        ax.set_title(f'Width {width}', fontsize=14, fontweight='bold')
        ax.set_xticks([i + 0.25 for i in range(3)])
        ax.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3'])
        ax.legend(fontsize=10)
        ax.grid(False)
        ax.set_ylim(0, 100)
    
    fig.suptitle('Sample Exit Distribution Across Layers\nby Network Width and Confidence Multiplier (n=5 runs)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = Path(output_dir) / 'exit_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_power_breakdown(data: Dict[int, Dict], output_dir: str = "output"):
    """Plot 4: Power consumption breakdown by component with error bars."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    multiplier_target = 3.0  # Show breakdown for best performing multiplier
    width_labels = []
    cpu_gpu_power = []
    soc_power = []
    cpu_gpu_errors = []
    soc_errors = []
    
    for width in sorted(data.keys()):
        width_labels.append(f'Width {width}')
        
        # Find the target multiplier result
        for result in data[width]['energy_analysis']['multiplier_results']:
            if result['multiplier'] == multiplier_target:
                # Average power across all exit layers weighted by samples
                total_samples = sum(exit_info['samples_mean'] for exit_info in result['exit_distribution'])
                
                weighted_cpu_gpu = sum(
                    exit_info['cpu_gpu_power_w_mean'] * exit_info['samples_mean'] 
                    for exit_info in result['exit_distribution']
                ) / total_samples
                
                weighted_soc = sum(
                    exit_info['soc_power_w_mean'] * exit_info['samples_mean']
                    for exit_info in result['exit_distribution']
                ) / total_samples
                
                cpu_gpu_power.append(weighted_cpu_gpu)
                soc_power.append(weighted_soc)
                # Note: We don't have std for power components in aggregated data
                # Could be added if needed
                cpu_gpu_errors.append(0)
                soc_errors.append(0)
                break
    
    x = np.arange(len(width_labels))
    width = 0.5
    
    p1 = ax.bar(x, cpu_gpu_power, width, label='CPU+GPU', color='#3498db', alpha=0.8)
    p2 = ax.bar(x, soc_power, width, bottom=cpu_gpu_power, label='SOC', color='#e74c3c', alpha=0.8)
    
    # Add value labels
    for i, (cpu, soc) in enumerate(zip(cpu_gpu_power, soc_power)):
        ax.text(i, cpu/2, f'{cpu:.2f}W', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        ax.text(i, cpu + soc/2, f'{soc:.2f}W', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(i, cpu + soc + 0.15, f'{cpu+soc:.2f}W', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Network Width', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Power (W)', fontsize=14, fontweight='bold')
    ax.set_title(f'Power Consumption Breakdown (Multiplier {multiplier_target}×)\nAcross Network Widths (n=5 runs)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(width_labels)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(False)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'power_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_energy_savings_highlight(data: Dict[int, Dict], output_dir: str = "output"):
    """Plot highlighting energy savings at 3.0× multiplier vs baseline."""
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    
    widths = sorted(data.keys())
    width_labels = [f'{w}' for w in widths]  # Just the numbers, no "Width" prefix
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # === Energy Comparison (Baseline vs 3.0×) ===
    baseline_energy = []
    baseline_energy_std = []
    multiplier_3x_energy = []
    multiplier_3x_energy_std = []
    
    for width in widths:
        # Baseline energy
        baseline_energy.append(data[width]['energy_analysis']['baseline']['per_sample_mj_mean'])
        baseline_energy_std.append(data[width]['energy_analysis']['baseline']['per_sample_mj_std'])
        
        # 3.0× multiplier energy
        for result in data[width]['energy_analysis']['multiplier_results']:
            if result['multiplier'] == 3.0:
                multiplier_3x_energy.append(result['per_sample_mj_mean'])
                multiplier_3x_energy_std.append(result['per_sample_mj_std'])
                break
    
    x = np.arange(len(widths))
    width_bar = 0.35
    
    bars1 = ax1.bar(x - width_bar/2, baseline_energy, width_bar, 
                    label='Full Network', 
                    color='#95a5a6', alpha=0.8, 
                    yerr=baseline_energy_std, capsize=5, error_kw={'linewidth': 2})
    
    bars2 = ax1.bar(x + width_bar/2, multiplier_3x_energy, width_bar, 
                    label='Early Exit', 
                    color='#27ae60', alpha=0.9, 
                    yerr=multiplier_3x_energy_std, capsize=5, error_kw={'linewidth': 2})
    
    # Get accuracy values for each width at baseline (1.0×) and 3.0×
    baseline_acc_for_energy = []
    multiplier_3x_acc_for_energy = []
    
    for width in widths:
        # Baseline accuracy (1.0× multiplier)
        for result in data[width]['energy_analysis']['multiplier_results']:
            if result['multiplier'] == 1.0:
                baseline_acc_for_energy.append(result['accuracy_mean'])
                break
        
        # 3.0× multiplier accuracy
        for result in data[width]['energy_analysis']['multiplier_results']:
            if result['multiplier'] == 3.0:
                multiplier_3x_acc_for_energy.append(result['accuracy_mean'])
                break
    
    ax1.set_xlabel('Network Width', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Energy per Sample (mJ)', fontsize=14, fontweight='bold', color='black')
    ax1.set_title('Energy Savings with Early Exit\n(n=5 runs)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(width_labels, fontsize=12)
    ax1.legend(fontsize=11, loc='lower left', framealpha=0.95)
    ax1.set_ylim(0, 19)
    ax1.tick_params(axis='y', labelcolor='black')
    
    # === Add second y-axis for accuracy ===
    ax2 = ax1.twinx()
    
    # Use blue color for all accuracy lines
    accuracy_color = '#1f77b4'  # Blue
    
    # Plot accuracy lines connecting Full Network to Early Exit for each width
    for i, width in enumerate(widths):
        # Plot line connecting baseline to early exit accuracy
        ax2.plot([x[i] - width_bar/2, x[i] + width_bar/2], 
                [baseline_acc_for_energy[i], multiplier_3x_acc_for_energy[i]], 
                color=accuracy_color, linewidth=2.5, marker='o', markersize=8, 
                zorder=10)
    
    ax2.set_ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold', color=accuracy_color)
    ax2.tick_params(axis='y', labelcolor=accuracy_color)
    ax2.set_ylim(90, 100)
    
    # Check if latency data is available
    has_latency = 'has_latency' in data[widths[0]]['energy_analysis']['baseline']
    
    # Get latency values if available
    baseline_latency = []
    baseline_latency_std = []
    multiplier_3x_latency = []
    multiplier_3x_latency_std = []
    
    if has_latency:
        for width in widths:
            # Baseline latency (from actual baseline - multiplier 1.0, all samples through all layers)
            baseline_latency.append(data[width]['energy_analysis']['baseline'].get('average_latency_ms_mean', 0))
            baseline_latency_std.append(data[width]['energy_analysis']['baseline'].get('average_latency_ms_std', 0))
            
            # 3.0× multiplier latency
            for result in data[width]['energy_analysis']['multiplier_results']:
                if result['multiplier'] == 3.0:
                    multiplier_3x_latency.append(result.get('average_latency_ms_mean', 0))
                    multiplier_3x_latency_std.append(result.get('average_latency_ms_std', 0))
                    break
        
        # === Add third y-axis for latency ===
        ax3 = ax1.twinx()
        
        # Offset the third axis to the right
        ax3.spines['right'].set_position(('outward', 60))
        
        # Use orange color for latency
        latency_color = '#ff7f0e'  # Orange
        
        # Plot latency lines connecting Full Network to Early Exit for each width
        for i, width in enumerate(widths):
            # Plot the line
            ax3.plot([x[i] - width_bar/2, x[i] + width_bar/2], 
                    [baseline_latency[i], multiplier_3x_latency[i]], 
                    color=latency_color, linewidth=2.5, marker='s', markersize=8, 
                    linestyle='--', zorder=10)
            
            # Add error bars at each endpoint
            # Baseline error bar
            ax3.errorbar(x[i] - width_bar/2, baseline_latency[i], 
                        yerr=baseline_latency_std[i],
                        fmt='none', ecolor=latency_color, elinewidth=2, capsize=5, capthick=2,
                        zorder=10)
            
            # Early exit error bar
            ax3.errorbar(x[i] + width_bar/2, multiplier_3x_latency[i], 
                        yerr=multiplier_3x_latency_std[i],
                        fmt='none', ecolor=latency_color, elinewidth=2, capsize=5, capthick=2,
                        zorder=10)
        
        ax3.set_ylabel('Average Latency (ms)', fontsize=14, fontweight='bold', color=latency_color)
        ax3.tick_params(axis='y', labelcolor=latency_color)
        
        # Set latency axis limits to accommodate baseline (~103-110 ms) and early exit (~10-20 ms)
        ax3.set_ylim(0, 150)
    
    # Add percentage changes above each network width
    for i, (baseline_e, mult_3x_e, baseline_a, mult_3x_a) in enumerate(zip(
        baseline_energy, multiplier_3x_energy, baseline_acc_for_energy, multiplier_3x_acc_for_energy)):
        
        # Calculate percentage changes
        energy_change = ((baseline_e - mult_3x_e) / baseline_e) * 100
        accuracy_change = mult_3x_a - baseline_a
        
        # Position above the bars
        y_pos = max(baseline_e, mult_3x_e) + 0.5
        
        # Build annotation text
        if has_latency and len(baseline_latency) > i and len(multiplier_3x_latency) > i:
            latency_change = ((baseline_latency[i] - multiplier_3x_latency[i]) / baseline_latency[i]) * 100
            annotation_text = (f'Energy: −{energy_change:.1f}%\n'
                             f'Accuracy: +{accuracy_change:.2f}%\n'
                             f'Latency: −{latency_change:.1f}%')
        else:
            annotation_text = f'Energy: −{energy_change:.1f}%\nAccuracy: +{accuracy_change:.2f}%'
        
        # Add combined annotation
        ax1.text(i, y_pos, annotation_text,
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.grid(False)
    ax1.grid(False)
    ax2.grid(False)
    if has_latency:
        ax3.grid(False)
    output_path = Path(output_dir) / 'energy_savings_highlight.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_progressive_multiplier_impact(data: Dict[int, Dict], output_dir: str = "output"):
    """Plot showing progressive impact of increasing multiplier values."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    multipliers = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    # === TOP PLOT: Energy Reduction Progress ===
    ax1 = axes[0]
    
    for width_idx, width in enumerate(sorted(data.keys())):
        baseline = data[width]['energy_analysis']['baseline']['per_sample_mj_mean']
        
        energy_values = []
        energy_reduction_pct = []
        
        for mult in multipliers:
            for result in data[width]['energy_analysis']['multiplier_results']:
                if result['multiplier'] == mult:
                    energy = result['per_sample_mj_mean']
                    energy_values.append(energy)
                    reduction = ((baseline - energy) / baseline) * 100
                    energy_reduction_pct.append(reduction)
                    break
        
        x_positions = np.arange(len(multipliers)) + width_idx * 0.25
        bars = ax1.bar(x_positions, energy_reduction_pct, width=0.23,
                      label=f'Width {width}',
                      color=['#2ecc71', '#3498db', '#e74c3c'][width_idx],
                      alpha=0.85)
        
        # Highlight 3.0× multiplier
        ax1.bar(x_positions[-1], energy_reduction_pct[-1], width=0.23,
               color=['#2ecc71', '#3498db', '#e74c3c'][width_idx],
               alpha=1.0, edgecolor='gold', linewidth=3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, energy_reduction_pct)):
            if i == len(bars) - 1:  # Highlight 3.0×
                ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
                        f'{val:.1f}%', ha='center', va='bottom',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
            elif val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom',
                        fontsize=9)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax1.axhline(y=20, color='green', linestyle='--', linewidth=2, alpha=0.7, label='20% Target')
    
    ax1.set_xlabel('Confidence Threshold Multiplier', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Energy Reduction (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Progressive Energy Savings with Increasing Confidence Threshold\n(Negative = Energy Increase)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(np.arange(len(multipliers)) + 0.25)
    ax1.set_xticklabels([f'{m}×' for m in multipliers], fontsize=11)
    ax1.legend(fontsize=10, loc='upper left', ncol=2)
    ax1.grid(False)
    
    # === BOTTOM PLOT: Accuracy Stability ===
    ax2 = axes[1]
    
    for width_idx, width in enumerate(sorted(data.keys())):
        accuracy_values = []
        accuracy_stds = []
        
        for mult in multipliers:
            for result in data[width]['energy_analysis']['multiplier_results']:
                if result['multiplier'] == mult:
                    accuracy_values.append(result['accuracy_mean'])
                    accuracy_stds.append(result['accuracy_std'])
                    break
        
        x_positions = np.arange(len(multipliers)) + width_idx * 0.25
        bars = ax2.bar(x_positions, accuracy_values, width=0.23,
                      label=f'Width {width}',
                      color=['#2ecc71', '#3498db', '#e74c3c'][width_idx],
                      alpha=0.85,
                      yerr=accuracy_stds, capsize=3, error_kw={'linewidth': 1.5})
        
        # Highlight 3.0× multiplier
        ax2.bar(x_positions[-1], accuracy_values[-1], width=0.23,
               color=['#2ecc71', '#3498db', '#e74c3c'][width_idx],
               alpha=1.0, edgecolor='gold', linewidth=3,
               yerr=accuracy_stds[-1], capsize=3, error_kw={'linewidth': 1.5})
        
        # Add value labels for 3.0×
        ax2.text(x_positions[-1], accuracy_values[-1] + accuracy_stds[-1] + 0.1,
                f'{accuracy_values[-1]:.2f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
    
    # Add reference line at 94%
    ax2.axhline(y=94, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='94% Reference')
    
    ax2.set_xlabel('Confidence Threshold Multiplier', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Overall Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Accuracy Maintained Across Different Confidence Thresholds\n(Error bars show standard deviation across 5 runs)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(np.arange(len(multipliers)) + 0.25)
    ax2.set_xticklabels([f'{m}×' for m in multipliers], fontsize=11)
    ax2.legend(fontsize=10, loc='lower right', ncol=2)
    ax2.grid(False)
    ax2.set_ylim(92, 96)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'progressive_multiplier_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_multiplier_vs_energy_accuracy(data: Dict[int, Dict], output_dir: str = "output"):
    """Plot 5: Dual Y-axis plot showing energy and accuracy vs multiplier."""
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    colors = {'100': '#2ecc71', '200': '#3498db', '300': '#e74c3c'}
    markers = {'100': 'o', '200': 's', '300': '^'}
    
    # First Y-axis: Energy consumption
    ax1.set_xlabel('Confidence Threshold Multiplier', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Energy per Sample (mJ)', fontsize=14, fontweight='bold', color='#2c3e50')
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    
    # Second Y-axis: Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='#8e44ad')
    ax2.tick_params(axis='y', labelcolor='#8e44ad')
    
    multipliers = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Plot energy and accuracy for each width
    for width, width_data in sorted(data.items()):
        energies = []
        energy_errors = []
        accuracies = []
        accuracy_errors = []
        
        for result in width_data['energy_analysis']['multiplier_results']:
            if result['multiplier'] in multipliers:
                energies.append(result['per_sample_mj_mean'])
                energy_errors.append(result['per_sample_mj_std'])
                accuracies.append(result['accuracy_mean'])
                accuracy_errors.append(result['accuracy_std'])
        
        # Plot energy on left axis (solid lines)
        line1 = ax1.errorbar(multipliers, energies, yerr=energy_errors,
                             marker=markers[str(width)], 
                             color=colors[str(width)],
                             linewidth=2.5, 
                             markersize=10,
                             linestyle='-',
                             label=f'Width {width} (Energy)',
                             alpha=0.8,
                             capsize=5,
                             capthick=2)
        
        # Plot accuracy on right axis (dashed lines)
        line2 = ax2.errorbar(multipliers, accuracies, yerr=accuracy_errors,
                             marker=markers[str(width)], 
                             color=colors[str(width)],
                             linewidth=2.5, 
                             markersize=10,
                             linestyle='--',
                             label=f'Width {width} (Accuracy)',
                             alpha=0.8,
                             capsize=5,
                             capthick=2)
    
    # Add grid
    ax1.grid(False)
    
    # Set x-axis ticks
    ax1.set_xticks(multipliers)
    ax1.set_xticklabels([f'{m}×' for m in multipliers])
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Organize legend by width
    legend_lines = []
    legend_labels = []
    for width in sorted(data.keys()):
        # Add energy line
        for line, label in zip(lines1, labels1):
            if f'Width {width}' in label:
                legend_lines.append(line)
                legend_labels.append(label)
                break
        # Add accuracy line
        for line, label in zip(lines2, labels2):
            if f'Width {width}' in label:
                legend_lines.append(line)
                legend_labels.append(label)
                break
    
    ax1.legend(legend_lines, legend_labels, fontsize=11, loc='upper left', 
               framealpha=0.9, ncol=1)
    
    # Title
    plt.title('Energy Consumption and Accuracy vs Confidence Threshold Multiplier\n(n=5 runs)', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'multiplier_vs_energy_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_comprehensive_dashboard(data: Dict[int, Dict], output_dir: str = "output"):
    """Plot 5: Comprehensive dashboard with multiple metrics and error bars."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Energy vs Accuracy (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    colors = {'100': '#2ecc71', '200': '#3498db', '300': '#e74c3c'}
    
    for width, width_data in sorted(data.items()):
        energies = [r['per_sample_mj_mean'] for r in width_data['energy_analysis']['multiplier_results']]
        energy_errors = [r['per_sample_mj_std'] for r in width_data['energy_analysis']['multiplier_results']]
        accuracies = [r['accuracy_mean'] for r in width_data['energy_analysis']['multiplier_results']]
        accuracy_errors = [r['accuracy_std'] for r in width_data['energy_analysis']['multiplier_results']]
        
        ax1.errorbar(energies, accuracies, xerr=energy_errors, yerr=accuracy_errors,
                    fmt='o-', label=f'Width {width}', 
                    color=colors[str(width)], linewidth=2, markersize=8,
                    capsize=4, capthick=1.5)
    
    ax1.set_xlabel('Energy per Sample (mJ)', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Energy-Accuracy Trade-off', fontweight='bold', fontsize=13)
    ax1.legend()
    ax1.grid(False)
    
    # 2. Best Energy Savings (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    widths = sorted(data.keys())
    best_savings = []
    best_savings_err = []
    
    for w in widths:
        max_savings = max(r['energy_savings_pct_mean'] for r in data[w]['energy_analysis']['multiplier_results'])
        # Find corresponding std
        for r in data[w]['energy_analysis']['multiplier_results']:
            if r['energy_savings_pct_mean'] == max_savings:
                best_savings.append(max_savings)
                best_savings_err.append(r['energy_savings_pct_std'])
                break
    
    bars = ax2.bar([f'W{w}' for w in widths], best_savings, 
                   yerr=best_savings_err,
                   color=['#2ecc71', '#3498db', '#e74c3c'],
                   capsize=5)
    for bar, val in zip(bars, best_savings):
        ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    ax2.set_ylabel('Energy Savings (%)', fontweight='bold')
    ax2.set_title('Best Energy Savings', fontweight='bold', fontsize=13)
    ax2.grid(False)
    
    # 3. Exit Distribution Heatmap (middle row, spans 3 columns)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create heatmap data (using means)
    heatmap_data = []
    labels = []
    
    for width in sorted(data.keys()):
        for result in data[width]['energy_analysis']['multiplier_results']:
            if result['multiplier'] in [1.0, 2.0, 3.0]:
                row = []
                for layer in [1, 2, 3]:
                    for exit_info in result['exit_distribution']:
                        if exit_info['layer'] == layer:
                            total = data[width]['energy_analysis']['baseline']['total_samples']
                            row.append((exit_info['samples_mean'] / total) * 100)
                            break
                    else:
                        row.append(0)
                heatmap_data.append(row)
                labels.append(f"W{width}-{result['multiplier']}×")
    
    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3'])
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels)
    ax3.set_title('Sample Exit Distribution Heatmap (%)', fontweight='bold', fontsize=13)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(3):
            text = ax3.text(j, i, f'{heatmap_data[i][j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax3, label='% of Samples')
    
    # 4. Layer-wise Accuracy (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    for width in sorted(data.keys()):
        layers = [0, 1, 2]
        accuracies = [
            data[width]['layer_analysis']['fixed_layer_accuracy'][f'layer_{i}']['mean']
            for i in layers
        ]
        errors = [
            data[width]['layer_analysis']['fixed_layer_accuracy'][f'layer_{i}']['std']
            for i in layers
        ]
        ax4.errorbar(layers, accuracies, yerr=errors, fmt='o-', label=f'W{width}', 
                    color=colors[str(width)], linewidth=2, markersize=8,
                    capsize=4, capthick=1.5)
    ax4.set_xlabel('Layer', fontweight='bold')
    ax4.set_ylabel('Accuracy', fontweight='bold')
    ax4.set_title('Fixed Layer Accuracy', fontweight='bold', fontsize=13)
    ax4.legend()
    ax4.grid(False)
    
    # 5. Baseline Energy (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])
    baseline_energies = [
        data[w]['energy_analysis']['baseline']['per_sample_mj_mean']
        for w in widths
    ]
    baseline_errors = [
        data[w]['energy_analysis']['baseline']['per_sample_mj_std']
        for w in widths
    ]
    bars = ax5.bar([f'W{w}' for w in widths], baseline_energies,
                   yerr=baseline_errors,
                   color=['#2ecc71', '#3498db', '#e74c3c'],
                   capsize=5)
    for bar, val in zip(bars, baseline_energies):
        ax5.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold')
    ax5.set_ylabel('Energy (mJ)', fontweight='bold')
    ax5.set_title('Baseline Energy per Sample', fontweight='bold', fontsize=13)
    ax5.grid(False)
    
    # 6. Specialization Scores (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    for width in sorted(data.keys()):
        layers = [0, 1, 2]
        scores = [
            data[width]['specialization']['scores'][f'layer_{i}']['mean']
            for i in layers
        ]
        errors = [
            data[width]['specialization']['scores'][f'layer_{i}']['std']
            for i in layers
        ]
        ax6.errorbar(layers, scores, yerr=errors, fmt='o-', label=f'W{width}',
                    color=colors[str(width)], linewidth=2, markersize=8,
                    capsize=4, capthick=1.5)
    ax6.set_xlabel('Layer', fontweight='bold')
    ax6.set_ylabel('Specialization Score', fontweight='bold')
    ax6.set_title('Layer Specialization', fontweight='bold', fontsize=13)
    ax6.legend()
    ax6.grid(False)
    ax6.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    fig.suptitle('Comprehensive Early Exit Energy Analysis Dashboard (n=5 runs)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    output_path = Path(output_dir) / 'comprehensive_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_width100_comparison(data: Dict[int, Dict], output_dir: str = "output"):
    """Plot horizontal bar comparison for Width 100: Energy, Latency, and Accuracy."""
    if 100 not in data:
        print("⚠ Width 100 data not found")
        return
    
    width_100 = data[100]
    
    # Extract baseline and early exit (3.0x) values
    baseline_energy = width_100['energy_analysis']['baseline']['per_sample_mj_mean']
    baseline_energy_std = width_100['energy_analysis']['baseline']['per_sample_mj_std']
    baseline_latency = width_100['energy_analysis']['baseline']['average_latency_ms_mean']
    baseline_latency_std = width_100['energy_analysis']['baseline']['average_latency_ms_std']
    
    # Get accuracy from dataset results
    if 'dataset_results' in width_100 and 'test' in width_100['dataset_results']:
        baseline_accuracy = width_100['dataset_results']['test']['accuracy_mean'] * 100
        baseline_accuracy_std = width_100['dataset_results']['test']['accuracy_std'] * 100
    else:
        baseline_accuracy = 94.0  # fallback
        baseline_accuracy_std = 0.1
    
    # Find 3.0x multiplier results
    early_exit_energy = None
    early_exit_latency = None
    early_exit_accuracy = None
    
    for result in width_100['energy_analysis']['multiplier_results']:
        if result['multiplier'] == 3.0:
            early_exit_energy = result['per_sample_mj_mean']
            early_exit_energy_std = result['per_sample_mj_std']
            early_exit_latency = result['average_latency_ms_mean']
            early_exit_latency_std = result['average_latency_ms_std']
            early_exit_accuracy = result['accuracy_mean']
            early_exit_accuracy_std = result['accuracy_std']
            break
    
    if early_exit_energy is None:
        print("⚠ Early exit (3.0x) data not found for Width 100")
        return
    
    # Create figure with 3 rows, 1 column
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Colors
    color_baseline = '#95a5a6'
    color_early_exit = '#27ae60'
    
    # --- Plot 1: Energy (mJ) ---
    ax1 = axes[0]
    categories = ['Early Exit', 'Full Network']  # Reversed order - Full Network on top
    energy_values = [early_exit_energy, baseline_energy]  # Reversed order
    energy_errors = [early_exit_energy_std, baseline_energy_std]  # Reversed order
    colors = [color_early_exit, color_baseline]  # Reversed order
    
    bars1 = ax1.barh(categories, energy_values, xerr=energy_errors, 
                     color=colors, alpha=0.8, capsize=5, 
                     error_kw={'linewidth': 2},
                     label=['Early Exit', 'Full Network'])
    
    # Add value labels at the end of bars
    for i, (bar, val, err) in enumerate(zip(bars1, energy_values, energy_errors)):
        ax1.text(val + err + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f} mJ',
                va='center', ha='left', fontsize=30, fontweight='bold')
    
    # Remove axes, gridlines, and title
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([])
    ax1.grid(False)
    ax1.set_xlim(0, max(energy_values) + max(energy_errors) + 2)
    
    # --- Plot 2: Latency (ms) ---
    ax2 = axes[1]
    latency_values = [early_exit_latency, baseline_latency]  # Reversed order
    latency_errors = [early_exit_latency_std, baseline_latency_std]  # Reversed order
    
    bars2 = ax2.barh(categories, latency_values, xerr=latency_errors,
                     color=colors, alpha=0.8, capsize=5,
                     error_kw={'linewidth': 2})
    
    # Add value labels at the end of bars
    for i, (bar, val, err) in enumerate(zip(bars2, latency_values, latency_errors)):
        ax2.text(val + err + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.2f} ms',
                va='center', ha='left', fontsize=30, fontweight='bold')
    
    # Remove axes, gridlines, and title
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.set_xticks([])
    ax2.grid(False)
    ax2.set_xlim(0, max(latency_values) + max(latency_errors) + 10)
    
    # --- Plot 3: Accuracy (%) ---
    ax3 = axes[2]
    accuracy_values = [early_exit_accuracy, baseline_accuracy]  # Reversed order
    accuracy_errors = [early_exit_accuracy_std, baseline_accuracy_std]  # Reversed order
    
    bars3 = ax3.barh(categories, accuracy_values, xerr=accuracy_errors,
                     color=colors, alpha=0.8, capsize=5,
                     error_kw={'linewidth': 2})
    
    # Add value labels at the end of bars
    for i, (bar, val, err) in enumerate(zip(bars3, accuracy_values, accuracy_errors)):
        ax3.text(val + err + 0.15, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%',
                va='center', ha='left', fontsize=30, fontweight='bold')
    
    # Remove axes, gridlines, and title
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.set_xticks([])
    ax3.grid(False)
    ax3.set_xlim(90, 100)
    
    # Add legend at the bottom
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_baseline, alpha=0.8, label='Full Network'),
        Patch(facecolor=color_early_exit, alpha=0.8, label='Early Exit')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               fontsize=14, frameon=True, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'width100_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("ENERGY ANALYSIS VISUALIZATION")
    print("="*70)
    
    # Create output directory with current date
    current_date = datetime.now().strftime("%Y%m%d")
    output_dir = Path("output") / current_date
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n--- Loading JSON Files ---")
    raw_data = load_json_files("logs")
    
    if len(raw_data) < 2:
        print("⚠ Need at least 2 width files to generate comparisons")
        return
    
    print(f"\n✓ Loaded data for {len(raw_data)} network width configurations")
    
    # Aggregate runs
    print("\n--- Aggregating Statistics Across 5 Runs ---")
    data = {}
    for width, runs in raw_data.items():
        print(f"  Aggregating {len(runs)} runs for width {width}...")
        data[width] = aggregate_runs(runs)
    
    print(f"✓ Aggregated {sum(len(runs) for runs in raw_data.values())} total runs")
    
    # Generate all plots
    print("\n--- Generating Visualizations ---")
    plot_energy_accuracy_tradeoff(data, output_dir)
    plot_energy_savings_comparison(data, output_dir)
    plot_exit_distribution(data, output_dir)
    plot_power_breakdown(data, output_dir)
    plot_multiplier_vs_energy_accuracy(data, output_dir)
    plot_comprehensive_dashboard(data, output_dir)
    
    # NEW: Generate focused plots highlighting 3.0× multiplier benefits
    print("\n--- Generating Focused Energy Savings Plots ---")
    plot_energy_savings_highlight(data, output_dir)
    plot_progressive_multiplier_impact(data, output_dir)
    
    # NEW: Generate Width 100 specific comparison
    print("\n--- Generating Width 100 Comparison Plot ---")
    plot_width100_comparison(data, output_dir)
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📅 Date: {current_date}")
    print("="*70)
    print("\n📊 Key Plots for Demonstrating Energy Savings:")
    print("   • energy_savings_highlight.png - Direct baseline vs 3.0× comparison")
    print("   • progressive_multiplier_impact.png - Full multiplier progression")
    print("   • width100_comparison.png - Width 100 Energy/Latency/Accuracy comparison")
    print("="*70)

if __name__ == "__main__":
    main()
