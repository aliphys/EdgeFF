import yaml
from pathlib import Path
from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT / 'configs'


def _deep_merge(base, override):
    result = dict(base) if isinstance(base, dict) else {}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path):
    path = Path(path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(config_path, merge_common=True):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        candidate = PACKAGE_ROOT / config_path
        if candidate.exists():
            config_path = candidate
        else:
            candidate = Path.cwd() / config_path
            if candidate.exists():
                config_path = candidate
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _load_yaml(config_path)

    if merge_common:
        common_path = CONFIG_ROOT / 'common.yaml'
        if common_path.exists():
            common_config = _load_yaml(common_path)
            config = _deep_merge(common_config, config)

    return config


def load_env(env_path=None):
    if env_path is None:
        env_path = PACKAGE_ROOT.parent / '.env'
    env_path = Path(env_path)
    if env_path.exists():
        load_dotenv(env_path)
        return env_path
    return None
