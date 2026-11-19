"""Recursive YAML configuration, with paths relative to the project root."""
from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parents[3]


def resolve_site_configuration(configuration):
    """Resolve editable site settings once; saved recordings retain full values."""
    result=copy.deepcopy(configuration)
    reference=result.pop('construction_site_file',None)
    if reference:
        result['construction_site']=yaml.safe_load((ROOT/reference).read_text())
    motion_reference=result.pop('motion_profile_file',None)
    if motion_reference:
        profile=yaml.safe_load((ROOT/motion_reference).read_text())['generation']
        result.setdefault('obstacles',{}).setdefault('dynamic',{}).update(amplitude=profile['amplitude_m'],
            max_speed=profile['speed_range_m_s'][1],speed_range_m_s=profile['speed_range_m_s'])
    return result


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
    resolved=merge(load_config(path.parent / parent), data) if parent else data
    return resolve_site_configuration(resolved)
