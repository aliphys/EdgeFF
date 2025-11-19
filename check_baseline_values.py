#!/usr/bin/env python3
"""Check baseline values in JSON files."""

import json

widths = [100, 200, 300]

for width in widths:
    with open(f'logs/width{width}run1.json') as f:
        data = json.load(f)
    
    baseline = data['energy_analysis']['baseline']
    
    print(f'\nWidth {width} Baseline (multiplier 1.0):')
    print(f'  Energy: {baseline["per_sample_mj_with_comm"]:.2f} mJ')
    print(f'  Latency: {baseline["average_latency_ms"]:.2f} ms')
    print(f'  Comm Energy: {baseline["communication_energy_mj"]:.4f} mJ')
    print(f'  Description: {baseline["description"]}')
