import sys
sys.path.append('.')
from visualize_energy_analysis import load_json_files, aggregate_runs

raw_data = load_json_files('logs')
data = {width: aggregate_runs(runs) for width, runs in raw_data.items()}

print('Width 100 Dataset Results:')
print('=' * 60)
print('\nTest Accuracy (Baseline):')
print(f'  Mean: {data[100]["dataset_results"]["test"]["accuracy_mean"]:.6f} ({data[100]["dataset_results"]["test"]["accuracy_mean"]*100:.4f}%)')
print(f'  Std:  {data[100]["dataset_results"]["test"]["accuracy_std"]:.10e} ({data[100]["dataset_results"]["test"]["accuracy_std"]*100:.10e}%)')

print(f'\nEarly Exit (3.0x) Accuracy:')
mult3 = next((r for r in data[100]['energy_analysis']['multiplier_results'] if r['multiplier'] == 3.0), None)
print(f'  Mean: {mult3["accuracy_mean"]:.6f} ({mult3["accuracy_mean"]:.4f}%)')
print(f'  Std:  {mult3["accuracy_std"]:.10e} ({mult3["accuracy_std"]:.10e}%)')
