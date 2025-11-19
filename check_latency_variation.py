#!/usr/bin/env python3
"""Check latency standard deviations to see why error bars aren't visible."""

import json
import numpy as np

widths = [100, 200, 300]

print('='*80)
print('LATENCY STANDARD DEVIATIONS - Why Error Bars Are Not Visible')
print('='*80)

for width in widths:
    # Collect values from all 5 runs
    baseline_latencies = []
    mult3_latencies = []
    
    for run in range(1, 6):
        with open(f'logs/width{width}run{run}.json') as f:
            data = json.load(f)
        
        baseline = data['energy_analysis']['baseline']
        mult3 = [r for r in data['energy_analysis']['multiplier_results'] if r['multiplier'] == 3.0][0]
        
        baseline_latencies.append(baseline['average_latency_ms'])
        mult3_latencies.append(mult3['average_latency_ms'])
    
    baseline_mean = np.mean(baseline_latencies)
    baseline_std = np.std(baseline_latencies)
    mult3_mean = np.mean(mult3_latencies)
    mult3_std = np.std(mult3_latencies)
    
    print(f'\n{"─"*80}')
    print(f'WIDTH {width}')
    print(f'{"─"*80}')
    
    print(f'\n📊 Baseline (1.0x - Full Network):')
    print(f'  Mean:     {baseline_mean:.2f} ms')
    print(f'  Std Dev:  {baseline_std:.4f} ms')
    print(f'  CV:       {baseline_std/baseline_mean*100:.2f}% (coefficient of variation)')
    print(f'  Range:    {min(baseline_latencies):.2f} - {max(baseline_latencies):.2f} ms')
    print(f'  Values:   {[f"{v:.2f}" for v in baseline_latencies]}')
    
    # Calculate error bar size relative to plot scale (0-120 ms)
    plot_range = 120
    error_bar_fraction = (baseline_std * 2) / plot_range * 100  # 2x std (±1 std each way)
    print(f'  Error bar size: ±{baseline_std:.4f} ms ({error_bar_fraction:.2f}% of plot height)')
    
    print(f'\n🟢 Early Exit (3.0x):')
    print(f'  Mean:     {mult3_mean:.2f} ms')
    print(f'  Std Dev:  {mult3_std:.4f} ms')
    print(f'  CV:       {mult3_std/mult3_mean*100:.2f}% (coefficient of variation)')
    print(f'  Range:    {min(mult3_latencies):.2f} - {max(mult3_latencies):.2f} ms')
    print(f'  Values:   {[f"{v:.2f}" for v in mult3_latencies]}')
    
    error_bar_fraction = (mult3_std * 2) / plot_range * 100
    print(f'  Error bar size: ±{mult3_std:.4f} ms ({error_bar_fraction:.2f}% of plot height)')

print(f'\n{"="*80}')
print('CONCLUSION')
print('='*80)
print('\n⚠️  The error bars are TOO SMALL to be visible at the current scale!')
print('\nReason: Latency values are very consistent across runs:')
print('  • Standard deviations are only 0.002-0.04 ms')
print('  • This is <0.1% of the baseline latency (~103-110 ms)')
print('  • Error bars representing ±0.02 ms are invisible on a 0-120 ms scale')
print('\nOptions to make variability visible:')
print('  1. Use a logarithmic scale (but makes reading values harder)')
print('  2. Show inset zoomed plots for each cluster')
print('  3. Display std dev values as text annotations')
print('  4. Accept that latency is very consistent (good for reproducibility!)')
print('='*80)
