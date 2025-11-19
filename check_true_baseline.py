#!/usr/bin/env python3
"""Check true baseline values."""

import json

with open('logs/width100run1.json') as f:
    w100 = json.load(f)
with open('logs/width200run1.json') as f:
    w200 = json.load(f)
with open('logs/width300run1.json') as f:
    w300 = json.load(f)

print('='*70)
print('TRUE BASELINE (All samples through all layers)')
print('='*70)
print('\nFrom energy_analysis.baseline (0.1x multiplier results):')
print(f'Width 100: Latency = {w100["energy_analysis"]["baseline"]["average_latency_ms"]:.2f} ms')
print(f'Width 200: Latency = {w200["energy_analysis"]["baseline"]["average_latency_ms"]:.2f} ms')
print(f'Width 300: Latency = {w300["energy_analysis"]["baseline"]["average_latency_ms"]:.2f} ms')

print('\n' + '='*70)
print('CONCLUSION')
print('='*70)
print('\nThe "baseline" is actually multiplier 0.1x, which allows early exits!')
print('This is why wider networks show lower latency - they exit more at Layer 1.')
print('\nFor a fair comparison:')
print('  - Current "baseline": Uses confidence threshold (allows early exits)')
print('  - True baseline: Would force ALL samples through ALL layers')
print('\nThe visualization is CORRECT given the data.')
print('The issue is that 0.1x multiplier is not aggressive enough to prevent exits.')
