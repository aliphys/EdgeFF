import json

# Load one JSON file with communication power
with open('logs/width100run1.json', 'r') as f:
    data = json.load(f)

print("=" * 70)
print("COMMUNICATION POWER VERIFICATION (Width 100)")
print("=" * 70)

# Show communication config
if 'communication_config' in data:
    config = data['communication_config']
    print("\nCommunication Configuration:")
    print(f"  Transmit Power: {config['transmit_power_w']} W")
    print(f"  Transmit Current: {config['transmit_current_ma']} mA @ {config['voltage_v']} V")
    print(f"  Data Rate: {config['data_rate_mbps']} Mbps")
    print(f"  Bytes per Activation: {config['bytes_per_activation']}")
    print(f"  Width 100 - Layer 1 output: {config['activation_sizes_bytes']['width_100']['layer_1']} bytes")
    print(f"  Width 100 - Layer 2 output: {config['activation_sizes_bytes']['width_100']['layer_2']} bytes")

# Show results for different multipliers
print("\n" + "=" * 70)
print("ENERGY WITH COMMUNICATION POWER")
print("=" * 70)

multipliers_to_show = [0.1, 1.0, 3.0]

for mult in multipliers_to_show:
    result = [r for r in data['energy_analysis']['multiplier_results'] if r['multiplier'] == mult][0]
    
    print(f"\nMultiplier {mult}x:")
    print(f"  Compute Energy:       {result['per_sample_mj']:.4f} mJ")
    print(f"  Communication Energy: {result.get('communication_energy_mj', 0):.4f} mJ")
    print(f"  Total with Comm:      {result.get('per_sample_mj_with_comm', result['per_sample_mj']):.4f} mJ")
    print(f"  Accuracy:             {result['overall_accuracy']:.2f}%")
    
    # Show exit distribution
    if 'exit_distribution' in result:
        print("  Exit Distribution:")
        total_samples = sum(exit_info['samples'] for exit_info in result['exit_distribution'])
        for exit_info in result['exit_distribution']:
            layer = exit_info['layer']
            samples = exit_info['samples']
            pct = (samples / total_samples * 100) if total_samples > 0 else 0
            comm = exit_info.get('communication_energy_mj', 0)
            print(f"    Layer {layer}: {samples:5d} samples ({pct:5.2f}%), comm: {comm:.4f} mJ")

# Calculate savings
baseline = [r for r in data['energy_analysis']['multiplier_results'] if r['multiplier'] == 0.1][0]
mult3 = [r for r in data['energy_analysis']['multiplier_results'] if r['multiplier'] == 3.0][0]

baseline_total = baseline.get('per_sample_mj_with_comm', baseline['per_sample_mj'])
mult3_total = mult3.get('per_sample_mj_with_comm', mult3['per_sample_mj'])

savings_pct = ((baseline_total - mult3_total) / baseline_total) * 100

print("\n" + "=" * 70)
print("ENERGY SAVINGS WITH COMMUNICATION POWER")
print("=" * 70)
print(f"Baseline (0.1x):  {baseline_total:.4f} mJ")
print(f"Early Exit (3.0x): {mult3_total:.4f} mJ")
print(f"Savings:          {savings_pct:.2f}%")
print(f"Accuracy change:  {mult3['overall_accuracy'] - baseline['overall_accuracy']:.2f}%")
