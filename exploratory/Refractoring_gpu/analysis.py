"""
analysis.py - Visualization and Analysis of W&B Experiment Data
===============================================================

This script generates visualization plots from Weights & Biases experiment
data, including latency, energy, power, and memory heatmaps.

Usage:
    python analysis.py --config analysis_config.yaml

Generated Plots:
    - latency_vs_width.png: Latency per sample vs network width (line plot)
    - energy_vs_width.png: Energy per sample vs network width (line plot)
    - latency_heatmap.png: Latency heatmap (width × batch_size)
    - energy_heatmap.png: Energy heatmap (width × batch_size)
    - power_heatmap.png: Power heatmap (width × batch_size)
    - memory_heatmap.png: Memory usage heatmap (width × batch_size)

Configuration (analysis_config.yaml):
    project: edgeff-network-width    # W&B project name
    sweep_id: <sweep_id>              # Source sweep for analysis

Requirements:
    - wandb API access (WANDB_API_KEY in .env)
    - matplotlib, pandas, numpy
"""

import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Analyze inference metrics from wandb')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Load .env
    root_dir = Path(__file__).resolve().parent.parent.parent
    dotenv_path = root_dir / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

    # Query wandb
    api = wandb.Api()
    project_name = config.get('project', 'edgeff-network-width')
    
    # Get latest eval run
    runs = api.runs(project_name, filters={'jobType': 'eval'}, order='-created_at')
    runs_list = list(runs)
    if not runs_list:
        print("No eval runs found")
        return
    latest_run = runs_list[0]
    print(f"Using latest eval run: {latest_run.id} ({latest_run.name})")

    data = []
    for row in latest_run.scan_history():
        if 'width' in row and 'batch_size' in row:
            data.append({
                'width': row.get('width'),
                'batch_size': row.get('batch_size'),
                'latency_per_sample_ms': row.get('latency_per_sample_ms', 0),
                'energy_per_sample_mj': row.get('energy_per_sample_mj', 0),
                'avg_power_mw': row.get('avg_power_mw', 0),
                'memory_mb': row.get('memory_mb', 0),
                'accuracy': row.get('accuracy', 0)
            })

    df = pd.DataFrame(data)
    if df.empty:
        print("No data found")
        return

    # (a) Latency vs width
    plt.figure(figsize=(10, 6))
    for bs in df['batch_size'].unique():
        subset = df[df['batch_size'] == bs]
        plt.plot(subset['width'], subset['latency_per_sample_ms'], label=f'Batch {bs}', marker='o')
    plt.xlabel('Network Width')
    plt.ylabel('Latency per Sample (ms)')
    plt.title('Latency per Sample vs Network Width')
    plt.legend()
    plt.savefig('latency_vs_width.png')
    plt.show()

    # (b) Energy vs width
    plt.figure(figsize=(10, 6))
    for bs in df['batch_size'].unique():
        subset = df[df['batch_size'] == bs]
        plt.plot(subset['width'], subset['energy_per_sample_mj'], label=f'Batch {bs}', marker='o')
    plt.xlabel('Network Width')
    plt.ylabel('Energy per Sample (mJ)')
    plt.title('Energy per Sample vs Network Width')
    plt.legend()
    plt.savefig('energy_vs_width.png')
    plt.show()

    # (c) Heatmap latency vs width & batch_size
    pivot_lat = df.pivot_table(values='latency_per_sample_ms', index='width', columns='batch_size', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_lat, cmap='viridis', aspect='auto')
    plt.colorbar(label='Latency per Sample (ms)')
    plt.xticks(range(len(pivot_lat.columns)), pivot_lat.columns)
    plt.yticks(range(len(pivot_lat.index)), pivot_lat.index)
    plt.xlabel('Batch Size')
    plt.ylabel('Network Width')
    plt.title('Latency Heatmap')
    plt.savefig('latency_heatmap.png')
    plt.show()

    # (d) Heatmap energy vs width & batch_size
    pivot_eng = df.pivot_table(values='energy_per_sample_mj', index='width', columns='batch_size', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_eng, cmap='plasma', aspect='auto')
    plt.colorbar(label='Energy per Sample (mJ)')
    plt.xticks(range(len(pivot_eng.columns)), pivot_eng.columns)
    plt.yticks(range(len(pivot_eng.index)), pivot_eng.index)
    plt.xlabel('Batch Size')
    plt.ylabel('Network Width')
    plt.title('Energy Heatmap')
    plt.savefig('energy_heatmap.png')
    plt.show()

    # (e) Memory heatmap
    if 'memory_mb' in df.columns:
        pivot_mem = df.pivot_table(values='memory_mb', index='width', columns='batch_size', aggfunc='mean')
        plt.figure(figsize=(10, 6))
        plt.imshow(pivot_mem, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Memory (MB)')
        plt.xticks(range(len(pivot_mem.columns)), pivot_mem.columns)
        plt.yticks(range(len(pivot_mem.index)), pivot_mem.index)
        plt.xlabel('Batch Size')
        plt.ylabel('Network Width')
        plt.title('PyTorch Memory Heatmap')
        plt.savefig('memory_heatmap.png')
        plt.show()

    # (f) Power heatmap (assuming avg_power_mw is the power)
    if 'avg_power_mw' in df.columns:
        pivot_pow = df.pivot_table(values='avg_power_mw', index='width', columns='batch_size', aggfunc='mean')
        plt.figure(figsize=(10, 6))
        plt.imshow(pivot_pow, cmap='hot', aspect='auto')
        plt.colorbar(label='Power (mW)')
        plt.xticks(range(len(pivot_pow.columns)), pivot_pow.columns)
        plt.yticks(range(len(pivot_pow.index)), pivot_pow.index)
        plt.xlabel('Batch Size')
        plt.ylabel('Network Width')
        plt.title('Power Consumption Heatmap')
        plt.savefig('power_heatmap.png')
        plt.show()

if __name__ == '__main__':
    main()