import yaml
import os
from types import SimpleNamespace
import argparse
from typing import Any, Dict, Union
import torch

def load_config(path="configs/default.yaml", model=None, data=None, overrides=None, resume_path=None):
    if resume_path:
        # Load flat snapshot config
        resume_config_path = os.path.join(resume_path, "config.yaml")
        with open(resume_config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg = dict_to_namespace(cfg)
        cfg = apply_missing_defaults(cfg)
        return cfg

    # Load base/default config
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Determine active model/data keys
    model_key = model or cfg.get("defaults", {}).get("model", "base")
    data_key = data or cfg.get("defaults", {}).get("data", "demo")

    # Load selected model config file
    model_path = os.path.join("configs", "model", f"{model_key}.yaml")
    with open(model_path, "r") as f:
        cfg["model"] = yaml.safe_load(f)

    # Load selected data config file
    data_path = os.path.join("configs", "data", f"{data_key}.yaml")
    with open(data_path, "r") as f:
        cfg["data"] = yaml.safe_load(f)

    # Apply CLI overrides
    if overrides:
        if overrides.train_lr is not None:
            cfg.setdefault("train", {})["lr"] = overrides.train_lr
        if overrides.model_hidden_dim is not None:
            cfg.setdefault("model", {})["hidden_dim"] = overrides.model_hidden_dim
        if overrides.model_features is not None:
            cfg.setdefault("model", {})["features"] = overrides.model_features
        if overrides.model_fc_bottleneck is not None:
            cfg.setdefault("model", {})["fc_bottleneck"] = overrides.model_fc_bottleneck
        if overrides.model_fc_recurrent is not None:
            cfg.setdefault("model", {})["fc_recurrent"] = overrides.model_fc_recurrent
        if overrides.model_conv_recurrent is not None:
            cfg.setdefault("model", {})["conv_recurrent"] = overrides.model_conv_recurrent
        if overrides.model_soft_reset is not None:
            cfg.setdefault("model", {})["soft_reset"] = overrides.model_soft_reset
        if overrides.model_skip_connections is not None:
            cfg.setdefault("model", {})["skip_connections"] = overrides.model_skip_connections
        if overrides.model_analog is not None:
            cfg.setdefault("model", {})["analog"] = overrides.model_analog
        if overrides.data_use_static is not None:
            cfg.setdefault("data", {})["static"] = overrides.data_use_static
        if overrides.data_used_T is not None:
            cfg.setdefault("data", {})["used_T"] = overrides.data_used_T
        if overrides.model_initial_scaling is not None:
            cfg.setdefault("model", {})["initial_scaling"] = overrides.model_initial_scaling
        if overrides.data_downscale is not None:
            cfg.setdefault("data", {})["downscale"] = overrides.data_downscale
        if overrides.cfg_description is not None:
            cfg["description"] = overrides.cfg_description

    cfg = dict_to_namespace(cfg)
    cfg = apply_missing_defaults(cfg)
    return cfg

def apply_missing_defaults(cfg):
    # Defaults for model section
    model_defaults = {
        "initial_scaling": 1,
        "conv_recurrent": False,
        "soft_reset": False,
        "skip_connections": True, 
        "analog": False,  # Default to False for non-analog skips
    }
    
    data_defaults = {
        "augmentation_intensity": 0.0,
        "static": False,  # Default to False for dynamic data
    }

    for key, value in model_defaults.items():
        if not hasattr(cfg.model, key):
            setattr(cfg.model, key, value)
            
    for key, value in data_defaults.items():
        if not hasattr(cfg.data, key):
            setattr(cfg.data, key, value)

    # Similarly, add defaults for other sections if needed
    return cfg

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