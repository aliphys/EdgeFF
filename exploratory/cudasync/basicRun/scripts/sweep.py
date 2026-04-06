#!/usr/bin/env python3
"""W&B sweep launcher for basicRun."""

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import yaml
import wandb
from dotenv import load_dotenv
from basicRun.config import load_env


def resolve_config(config_path):
    candidate = Path(config_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidate = PACKAGE_ROOT / config_path
    if candidate.exists():
        return candidate

    candidate = Path.cwd() / config_path
    if candidate.exists():
        return candidate

    return candidate


def main():
    parser = argparse.ArgumentParser(description='Launch a W&B sweep for basicRun.')
    parser.add_argument('--config', type=str, default='configs/sweep.yaml', help='Path to sweep config YAML file')
    parser.add_argument('--project', type=str, default=None, help='W&B project name')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity (username or team name)')
    parser.add_argument('--run-agent', action='store_true', help='Run the sweep agent immediately after creation')
    parser.add_argument('--count', type=int, default=None, help='Number of sweep runs to execute')
    args = parser.parse_args()

    load_env()
    config_path = resolve_config(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config file not found: {config_path}")

    with open(config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    project_name = args.project or sweep_config.get('project', 'edgeff-refactor')

    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name, entity=args.entity)
    print(f"Sweep created successfully: {sweep_id}")
    print(f"Project: {project_name}")

    if args.run_agent:
        print('Starting sweep agent...')
        os.environ['WANDB_AGENT_MAX_INITIAL_FAILURES'] = '1000'
        if args.count:
            wandb.agent(sweep_id, project=project_name, count=args.count)
        else:
            wandb.agent(sweep_id, project=project_name)


if __name__ == '__main__':
    main()
