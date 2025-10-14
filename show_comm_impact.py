#!/usr/bin/env python3
"""Show communication power impact on energy savings."""

import json

# Load width 100 run 1
with open('logs/width100run1.json') as f:
    w100 = json.load(f)

baseline = w100['energy_analysis']['baseline']
mult3 = [r for r in w100['energy_analysis']['multiplier_results'] if r['multiplier'] == 3.0][0]

print('='*60)
print('COMMUNICATION POWER IMPACT (Width 100)')
print('='*60)

print(f'\nBaseline (Multiplier 0.1x):')
print(f'  Compute Only:     {baseline["per_sample_mj"]:.4f} mJ')
print(f'  + Communication:  {baseline["communication_energy_mj"]:.4f} mJ')
print(f'  Total:            {baseline["per_sample_mj_with_comm"]:.4f} mJ')

print(f'\nEarly Exit (Multiplier 3.0x):')
print(f'  Compute Only:     {mult3["per_sample_mj"]:.4f} mJ')
print(f'  + Communication:  {mult3["communication_energy_mj"]:.4f} mJ')
print(f'  Total:            {mult3["per_sample_mj_with_comm"]:.4f} mJ')

compute_savings = (baseline['per_sample_mj'] - mult3['per_sample_mj']) / baseline['per_sample_mj'] * 100
total_savings = (baseline['per_sample_mj_with_comm'] - mult3['per_sample_mj_with_comm']) / baseline['per_sample_mj_with_comm'] * 100

print(f'\nEnergy Savings:')
print(f'  Compute Only:     {compute_savings:.2f}%')
print(f'  With Comm Power:  {total_savings:.2f}%')

comm_impact_baseline = baseline['communication_energy_mj'] / baseline['per_sample_mj'] * 100
comm_impact_mult3 = mult3['communication_energy_mj'] / mult3['per_sample_mj'] * 100

print(f'\nCommunication Overhead:')
print(f'  Baseline:         {comm_impact_baseline:.2f}% of compute')
print(f'  Early Exit:       {comm_impact_mult3:.2f}% of compute')

print('='*60)
print('\nKey Insight:')
print('Early exit reduces BOTH compute and communication energy!')
print(f'Communication savings: {(baseline["communication_energy_mj"] - mult3["communication_energy_mj"]) / baseline["communication_energy_mj"] * 100:.1f}%')
print('(More samples exit at Layer 1, avoiding transmission)')
print('='*60)
