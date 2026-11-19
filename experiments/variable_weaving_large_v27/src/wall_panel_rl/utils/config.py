"""Recursive YAML configuration, with paths relative to the project root."""
from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parents[3]


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and key in result
            else value
        )
    return result


def load_config(path=None):
    path = Path(path or ROOT / "configs/base.yaml").resolve()
    data = yaml.safe_load(path.read_text())
    parent = data.pop("extends", None)
    return merge(load_config(path.parent / parent), data) if parent else data
