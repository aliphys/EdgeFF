#!/usr/bin/env python3
"""Check energy values from JSON files to compare with plot."""

import json

widths = [100, 200, 300]

print('Energy Values from JSON (Run 1):')
print('='*70)

for width in widths:
    with open(f'logs/width{width}run1.json') as f:
        data = json.load(f)
    
    baseline = data['energy_analysis']['baseline']
    mult3 = [r for r in data['energy_analysis']['multiplier_results'] if r['multiplier'] == 3.0][0]
    
    print(f'\nWidth {width}:')
    print(f'  Baseline (1.0x - no early exit):')
    print(f'    Per-sample energy: {baseline["per_sample_mj"]:.3f} mJ (compute only)')
    print(f'    Communication:     {baseline["communication_energy_mj"]:.4f} mJ')
    print(f'    Total with comm:   {baseline["per_sample_mj_with_comm"]:.3f} mJ')
    
    print(f'  Early Exit (3.0x):')
    print(f'    Per-sample energy: {mult3["per_sample_mj"]:.3f} mJ (compute only)')
    print(f'    Communication:     {mult3["communication_energy_mj"]:.4f} mJ')
    print(f'    Total with comm:   {mult3["per_sample_mj_with_comm"]:.3f} mJ')
    
    energy_savings = (baseline["per_sample_mj_with_comm"] - mult3["per_sample_mj_with_comm"]) / baseline["per_sample_mj_with_comm"] * 100
    print(f'  Energy Savings:      {energy_savings:.1f}%')

print('\n' + '='*70)
print('\nAveraged across 5 runs:')
print('='*70)

# Check aggregated values
for width in widths:
    energy_sum_baseline = 0
    energy_sum_mult3 = 0
    
    for run in range(1, 6):
        with open(f'logs/width{width}run{run}.json') as f:
            data = json.load(f)
        baseline = data['energy_analysis']['baseline']
        mult3 = [r for r in data['energy_analysis']['multiplier_results'] if r['multiplier'] == 3.0][0]
        
        energy_sum_baseline += baseline["per_sample_mj_with_comm"]
        energy_sum_mult3 += mult3["per_sample_mj_with_comm"]
    
    avg_baseline = energy_sum_baseline / 5
    avg_mult3 = energy_sum_mult3 / 5
    avg_savings = (avg_baseline - avg_mult3) / avg_baseline * 100
    
    print(f'\nWidth {width}:')
    print(f'  Baseline average:    {avg_baseline:.2f} mJ')
    print(f'  Early Exit average:  {avg_mult3:.2f} mJ')
    print(f'  Energy Savings:      {avg_savings:.1f}%')
