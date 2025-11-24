#!/usr/bin/env python3
"""
Quick test to verify sweep setup is working correctly.
This script validates the configuration without running the full sweep.
"""
import yaml
from pathlib import Path

def main():
    print("=" * 60)
    print("W&B Sweep Configuration Validator")
    print("=" * 60)
    
    config_path = Path(__file__).parent / 'sweep_config.yaml'
    
    # Load and validate YAML
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"✓ Successfully loaded: {config_path}")
    except Exception as e:
        print(f"✗ Error loading YAML: {e}")
        return False
    
    # Validate required fields
    required_fields = ['program', 'method', 'metric', 'parameters']
    for field in required_fields:
        if field in config:
            print(f"✓ Found required field: {field}")
        else:
            print(f"✗ Missing required field: {field}")
            return False
    
    # Count total combinations
    params = config['parameters']
    total_runs = 1
    grid_params = []
    
    print("\n" + "=" * 60)
    print("Parameters Configuration:")
    print("=" * 60)
    
    for param_name, param_config in params.items():
        if 'values' in param_config:
            n_values = len(param_config['values'])
            total_runs *= n_values
            grid_params.append((param_name, n_values))
            print(f"  {param_name}: {n_values} values (grid search)")
            for val in param_config['values']:
                print(f"    - {val}")
        elif 'value' in param_config:
            print(f"  {param_name}: {param_config['value']} (fixed)")
        else:
            print(f"  {param_name}: {param_config}")
    
    print("\n" + "=" * 60)
    print("Sweep Summary:")
    print("=" * 60)
    print(f"Method: {config['method']}")
    print(f"Metric: {config['metric']['name']} ({config['metric']['goal']})")
    print(f"Total combinations: {total_runs} runs")
    
    if grid_params:
        print(f"\nGrid search over:")
        for param_name, n_values in grid_params:
            print(f"  - {param_name}: {n_values} values")
    
    print("\n" + "=" * 60)
    print("Validation Complete! ✓")
    print("=" * 60)
    print("\nTo run the sweep:")
    print(f"  python run_sweep.py --run-agent")
    print("\nOr create sweep first, then run agent:")
    print(f"  python run_sweep.py")
    print(f"  wandb agent <sweep-id>")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    main()
