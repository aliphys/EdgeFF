#!/usr/bin/env python3
"""Debug regex matching for baseline energy."""

import re

with open('logs/width100run1.log', 'r', encoding='utf-8', errors='ignore') as f:
    log_content = f.read()

# Current regex from parse_log_to_json.py
baseline_match = re.search(
    r'--- Baseline Energy Analysis \(multiplier = 1\.0\) ---.*?'
    r'Average energy per sample:\s*([\d.]+)\s*mJ.*?'
    r'Total energy consumed:\s*([\d.]+)\s*J',
    log_content, re.DOTALL
)

if baseline_match:
    print('✅ Regex matched!')
    print(f'Group 1 (per-sample): {baseline_match.group(1)} mJ')
    print(f'Group 2 (total): {baseline_match.group(2)} J')
    print(f'\nMatched text preview:')
    print(baseline_match.group(0)[:500])
else:
    print('❌ Regex did not match!')
