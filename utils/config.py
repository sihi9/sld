import yaml
from types import SimpleNamespace


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)


def dict_to_namespace(d):
    """Recursively convert a nested dict to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d
