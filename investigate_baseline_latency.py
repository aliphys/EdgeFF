#!/usr/bin/env python3
"""Investigate why width 300 has lower baseline latency than width 100."""

import json

# Load all three widths
with open('logs/width100run1.json') as f:
    w100 = json.load(f)
with open('logs/width200run1.json') as f:
    w200 = json.load(f)
with open('logs/width300run1.json') as f:
    w300 = json.load(f)

print('='*70)
print('BASELINE LATENCY INVESTIGATION (Multiplier 0.1x)')
print('='*70)

for name, data in [('Width 100', w100), ('Width 200', w200), ('Width 300', w300)]:
    baseline = [r for r in data['energy_analysis']['multiplier_results'] if r['multiplier'] == 0.1][0]
    
    print(f'\n{name}:')
    print(f'  Average Latency: {baseline["average_latency_ms"]:.2f} ms')
    print(f'\n  Exit Distribution:')
    print(f'  {"Layer":<7} {"Samples":<10} {"Pct":<8} {"Compute":<12} {"Upload":<12} {"RTT":<8} {"Total":<12}')
    print(f'  {"-"*75}')
    
    total_samples = sum(e['samples'] for e in baseline['exit_distribution'])
    
    for exit_info in baseline['exit_distribution']:
        layer = exit_info['layer']
        samples = exit_info['samples']
        pct = samples / total_samples * 100
        compute = exit_info['cumulative_time_ms']
        upload = exit_info['upload_time_ms']
        rtt = exit_info['network_rtt_ms']
        total = exit_info['total_latency_ms']
        
        print(f'  {layer:<7} {samples:<10} {pct:>6.2f}% {compute:>10.3f} ms {upload:>10.3f} ms {rtt:>6.0f} ms {total:>10.2f} ms')
    
    # Calculate weighted average manually
    weighted_latency = sum(e['total_latency_ms'] * e['samples'] for e in baseline['exit_distribution']) / total_samples
    print(f'\n  Manual Weighted Avg: {weighted_latency:.2f} ms')
    print(f'  JSON Average:        {baseline["average_latency_ms"]:.2f} ms')

print('\n' + '='*70)
print('KEY OBSERVATION')
print('='*70)
print('\nThe issue is that baseline (0.1x) has DIFFERENT exit distributions!')
print('Width 100: More samples exit at Layer 3 (highest latency)')
print('Width 300: More samples exit at Layer 1 (lowest latency)')
print('\nThis is INCORRECT for a baseline comparison!')
print('Baseline should force ALL samples through ALL layers.')
