#!/usr/bin/env python3
"""Extract baseline energy from log files to verify JSON parsing."""

import re

widths = [100, 200, 300]

print('Baseline Energy from LOG files (Run 1):')
print('='*70)

for width in widths:
    with open(f'logs/width{width}run1.log', 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    # Find the Baseline Energy Analysis section
    baseline_match = re.search(
        r'--- Baseline Energy Analysis \(multiplier = 1\.0\) ---.*?'
        r'Average energy per sample:\s*([\d.]+)\s*mJ.*?'
        r'Total energy consumed:\s*([\d.]+)\s*J',
        log_content, re.DOTALL
    )
    
    if baseline_match:
        energy_mj = float(baseline_match.group(1))
        total_j = float(baseline_match.group(2))
        print(f'\nWidth {width}:')
        print(f'  Per-sample energy: {energy_mj:.3f} mJ (from log)')
        print(f'  Total energy: {total_j:.3f} J (from log)')
    else:
        print(f'\nWidth {width}: Could not find baseline energy in log')
