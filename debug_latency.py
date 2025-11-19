#!/usr/bin/env python3
"""Debug script to test latency calculation."""

import json

# Read the JSON
with open('logs/width100run1.json') as f:
    data = json.load(f)

print("Checking add_latency_to_results execution...")
print(f"Has baseline: {'baseline' in data['energy_analysis']}")
print(f"Has multiplier_results: {'multiplier_results' in data['energy_analysis']}")
print(f"Number of multiplier_results: {len(data['energy_analysis']['multiplier_results'])}")

# Check if any result has Layer 3
for i, result in enumerate(data['energy_analysis']['multiplier_results']):
    print(f"\nResult {i} (multiplier {result['multiplier']}):")
    print(f"  Has exit_distribution: {'exit_distribution' in result}")
    if 'exit_distribution' in result:
        layers = [e['layer'] for e in result['exit_distribution']]
        print(f"  Layers: {layers}")
        for exit_info in result['exit_distribution']:
            if exit_info['layer'] == 3:
                print(f"  Layer 3 cumulative_time_ms: {exit_info.get('cumulative_time_ms', 'NOT FOUND')}")

print(f"\nBaseline keys: {list(data['energy_analysis']['baseline'].keys())}")
print(f"Has average_latency_ms in baseline: {'average_latency_ms' in data['energy_analysis']['baseline']}")
