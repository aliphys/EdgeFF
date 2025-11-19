#!/usr/bin/env python3
"""Verify latency calculations with network delays."""

import json

with open('logs/width100run1.json') as f:
    data = json.load(f)

# Check if latency is enabled
if 'latency_config' not in data:
    print("⚠ Latency calculations not enabled in JSON file")
    print("Run: python parse_log_to_json.py --enable-comm-power --enable-latency")
    exit(1)

config = data['latency_config']
print('='*70)
print('LATENCY VERIFICATION (Width 100)')
print('='*70)
print(f"\nLatency Configuration:")
print(f"  RTT: {config['rtt_ms']} ms")
print(f"  One-way latency: {config['one_way_latency_ms']} ms")
print(f"  Data rate: {config['data_rate_mbps']} Mbps")

print('\n' + '='*70)
print('LATENCY BREAKDOWN BY MULTIPLIER')
print('='*70)

for result in data['energy_analysis']['multiplier_results']:
    mult = result['multiplier']
    avg_latency = result.get('average_latency_ms', 0)
    
    print(f"\n📊 Multiplier {mult}x:")
    print(f"  Weighted Average Latency: {avg_latency:.2f} ms")
    print(f"\n  Per-Layer Breakdown:")
    
    total_samples = sum(exit_info['samples'] for exit_info in result['exit_distribution'])
    
    for exit_info in result['exit_distribution']:
        layer = exit_info['layer']
        samples = exit_info['samples']
        pct = (samples / total_samples * 100)
        
        compute_time = exit_info['cumulative_time_ms']
        upload_time = exit_info.get('upload_time_ms', 0)
        network_rtt = exit_info.get('network_rtt_ms', 0)
        total_lat = exit_info.get('total_latency_ms', compute_time)
        
        print(f"\n    Layer {layer}: {samples:5d} samples ({pct:5.2f}%)")
        print(f"      Compute Time:    {compute_time:8.3f} ms")
        if layer > 1:
            print(f"      Upload Time:     {upload_time:8.3f} ms")
            print(f"      Network RTT:     {network_rtt:8.1f} ms")
            print(f"      ───────────────────────────")
        print(f"      Total Latency:   {total_lat:8.2f} ms")
        print(f"      Contribution:    {(total_lat * pct / 100):8.2f} ms to avg")

# Calculate latency savings
baseline = [r for r in data['energy_analysis']['multiplier_results'] 
            if r['multiplier'] == 0.1][0]
mult1 = [r for r in data['energy_analysis']['multiplier_results'] 
         if r['multiplier'] == 1.0][0]
mult3 = [r for r in data['energy_analysis']['multiplier_results'] 
         if r['multiplier'] == 3.0][0]

baseline_lat = baseline.get('average_latency_ms', 0)
mult1_lat = mult1.get('average_latency_ms', 0)
mult3_lat = mult3.get('average_latency_ms', 0)

latency_savings_1x = (baseline_lat - mult1_lat) / baseline_lat * 100
latency_savings_3x = (baseline_lat - mult3_lat) / baseline_lat * 100

print('\n' + '='*70)
print('LATENCY SAVINGS')
print('='*70)
print(f"\nBaseline (0.1x):   {baseline_lat:7.2f} ms")
print(f"Multiplier 1.0x:   {mult1_lat:7.2f} ms  (savings: {latency_savings_1x:5.2f}%)")
print(f"Multiplier 3.0x:   {mult3_lat:7.2f} ms  (savings: {latency_savings_3x:5.2f}%)")

print('\n' + '='*70)
print('KEY INSIGHTS')
print('='*70)

# Calculate Layer 1 vs Cloud layer latency
layer1_exits_baseline = baseline['exit_distribution'][0]['samples']
layer1_exits_mult3 = mult3['exit_distribution'][0]['samples']
total_samples_baseline = sum(e['samples'] for e in baseline['exit_distribution'])
total_samples_mult3 = sum(e['samples'] for e in mult3['exit_distribution'])

layer1_pct_baseline = (layer1_exits_baseline / total_samples_baseline * 100)
layer1_pct_mult3 = (layer1_exits_mult3 / total_samples_mult3 * 100)

layer1_latency = baseline['exit_distribution'][0]['total_latency_ms']
cloud_latency_avg = sum(e['total_latency_ms'] for e in baseline['exit_distribution'][1:]) / 2

print(f"\n1. On-Device (Layer 1) Latency: {layer1_latency:.3f} ms")
print(f"   Cloud (Layer 2-3) Latency:   ~{cloud_latency_avg:.1f} ms")
print(f"   Network overhead: {(cloud_latency_avg / layer1_latency):.0f}× slower")

print(f"\n2. Layer 1 Exit Percentage:")
print(f"   Baseline (0.1x): {layer1_pct_baseline:.1f}%")
print(f"   Early Exit (3.0x): {layer1_pct_mult3:.1f}%")
print(f"   Increase: +{(layer1_pct_mult3 - layer1_pct_baseline):.1f}pp")

print(f"\n3. Early exit reduces latency by {latency_savings_3x:.1f}% by:")
print(f"   • Avoiding cloud round-trips for {layer1_pct_mult3:.1f}% of samples")
print(f"   • Each cloud inference takes ~{cloud_latency_avg:.1f} ms (vs {layer1_latency:.3f} ms local)")

print('='*70)
