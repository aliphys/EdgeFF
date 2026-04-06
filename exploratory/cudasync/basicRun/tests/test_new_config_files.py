#!/usr/bin/env python3
"""Smoke test for the new simplified basicRun YAML configs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from basicRun.config import load_config


def check_config(name):
    path = PROJECT_ROOT / 'configs' / f'{name}.yaml'
    config = load_config(path)
    print(f'Loaded {name}.yaml: project={config.get("project")}, keys={list(config.keys())}')
    assert config.get('project') == 'edgeff-refactor', f'{name}.yaml missing project'
    assert 'hw_interval_ms' in config, f'{name}.yaml missing hw_interval_ms after merge'
    return config


def main():
    print('=== config file smoke test ===')
    for cfg in ['run', 'eval', 'analysis', 'sweep']:
        check_config(cfg)
    print('\nSmoke test PASSED')


if __name__ == '__main__':
    main()
