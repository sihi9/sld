import yaml
from types import SimpleNamespace
import argparse
from typing import Any, Dict, Union
import torch

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



def namespace_to_dict(ns: Any) -> Any:
    """
    Recursively converts a SimpleNamespace (or nested structure) to a dict.

    Args:
        ns: SimpleNamespace or other object.

    Returns:
        A plain dict or the original value.
    """
    if isinstance(ns, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(ns).items()}
    elif isinstance(ns, dict):
        return {key: namespace_to_dict(value) for key, value in ns.items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(item) for item in ns]
    else:
        return ns



def serialize_config_for_logging(
    args: Union[argparse.Namespace, Dict],
    cfg: Any,
    markdown: bool = True
) -> str:
    """
    Combines CLI args and config into a single YAML-formatted string for logging.

    Args:
        args: CLI arguments (Namespace or dict).
        cfg:  Config object or dict.
        markdown: Wrap output in Markdown code block for TensorBoard.

    Returns:
        str: YAML-formatted string.
    """
    args_dict = vars(args) if isinstance(args, argparse.Namespace) else args
    cfg_dict_raw = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
    cfg_dict = namespace_to_dict(cfg_dict_raw)

    combined_config = {'args': args_dict, 'cfg': cfg_dict}
    yaml_str = yaml.dump(combined_config, sort_keys=False, default_flow_style=False)

    return f"```yaml\n{yaml_str}\n```" if markdown else yaml_str

def get_device(config_device="auto"):
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)