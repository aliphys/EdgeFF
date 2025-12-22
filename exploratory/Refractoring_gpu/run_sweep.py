#!/usr/bin/env python3
"""
W&B Sweep Launcher for Forward-Forward MNIST experiments
Usage:
    python run_sweep.py                  # Create sweep and print ID
    python run_sweep.py --run-agent      # Create sweep and run agent
    python run_sweep.py --count 5        # Limit to 5 runs
"""
import wandb
import yaml
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Launch W&B sweep for FF-MNIST')
    parser.add_argument('--config', type=str, default='sweep_config.yaml',
                        help='Path to sweep configuration YAML file')
    parser.add_argument('--project', type=str, default='replicateRun2000FF',
                        help='W&B project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='W&B entity (username or team name)')
    parser.add_argument('--run-agent', action='store_true',
                        help='Run the sweep agent immediately after creation')
    parser.add_argument('--count', type=int, default=None,
                        help='Number of sweep runs to execute (default: all combinations)')
    args = parser.parse_args()

    # Load sweep configuration
    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)
    
    print(f"Loaded sweep configuration from: {config_path}")
    print(f"Method: {sweep_config['method']}")
    print(f"Parameters: {list(sweep_config['parameters'].keys())}")
    
    # Initialize sweep
    sweep_kwargs = {
        'sweep': sweep_config,
        'project': args.project
    }
    if args.entity:
        sweep_kwargs['entity'] = args.entity
    
    sweep_id = wandb.sweep(**sweep_kwargs)
    
    print(f"\n{'='*60}")
    print(f"Sweep created successfully!")
    print(f"Sweep ID: {sweep_id}")
    print(f"{'='*60}")
    print(f"\nTo run the sweep agent, use:")
    print(f"  wandb agent {sweep_id}")
    print(f"\nOr run multiple agents in parallel:")
    print(f"  wandb agent {sweep_id}  # Terminal 1")
    print(f"  wandb agent {sweep_id}  # Terminal 2")
    print(f"\nView results at:")
    print(f"  https://wandb.ai/<your-entity>/{args.project}/sweeps/{sweep_id}")
    print(f"{'='*60}\n")
    
    # Optionally run agent
    if args.run_agent:
        print(f"Starting sweep agent...")
        # Set environment variable to ignore initial socket errors
        # This prevents the agent from shutting down due to transient network issues
        # Use a very high number instead of -1 to effectively disable failure detection
        os.environ['WANDB_AGENT_MAX_INITIAL_FAILURES'] = '1000'
        
        if args.count:
            print(f"Limiting to {args.count} runs")
            wandb.agent(sweep_id, project=args.project, count=args.count)
        else:
            print(f"Running all combinations")
            wandb.agent(sweep_id, project=args.project)

if __name__ == '__main__':
    main()
