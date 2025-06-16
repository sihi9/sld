import os
from datetime import datetime
import yaml
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple, Union

from utils.monitoring import SpikeLogger, count_neurons
from utils.config import serialize_config_for_logging, namespace_to_dict
from torchinfo import summary


def create_experiment_dir(cfg: Any, args: Namespace, base_dir: str = "experiments") -> str:
    """
    Creates a structured experiment directory with subfolders for logs and checkpoints.
    Saves config and arguments for reproducibility.

    Returns:
        str: Path to the root experiment directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg.model.name}_lr{cfg.train.lr}_{timestamp}"
    exp_dir = os.path.join(base_dir, exp_name)

    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    # Save configuration
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, 'w') as f:
        cfg_dict_raw = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
        cfg_dict = namespace_to_dict(cfg_dict_raw)
        yaml.dump(cfg_dict, f, default_flow_style=False)

    # Save CLI arguments
    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        f.write(str(args))

    # Save run metadata
    metadata = build_run_metadata(cfg)
    with open(os.path.join(exp_dir, "run_info.txt"), "w") as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    return exp_dir


def build_run_metadata(cfg: Any) -> Dict[str, Any]:
    """
    Generates standard metadata for the current experiment run.

    Args:
        cfg: The configuration object.

    Returns:
        Dict[str, Any]: Dictionary with run metadata.
    """
    return {
        "Started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Device": cfg.train.device,
        "AMP": cfg.train.amp,
        "Description": getattr(cfg, 'description', 'N/A'),
    }
    

class ExperimentManager:
    """
    Manages experiment folders, logging setup, and config serialization.
    """

    def __init__(self, cfg: Any, args: Namespace, base_dir: str = "experiments"):
        self.cfg = cfg
        self.args = args

        self.exp_dir: str = create_experiment_dir(cfg, args, base_dir)
        self.log_dir: str = os.path.join(self.exp_dir, "logs")
        self.checkpoint_dir: str = os.path.join(self.exp_dir, "checkpoints")

        self.logger = SpikeLogger(
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            vis_interval=cfg.log.vis_interval
        )

        self._log_config_to_tensorboard()
        self.log_run_metadata()


    def _log_config_to_tensorboard(self) -> None:
        """
        Logs the combined config and args to TensorBoard as YAML-formatted text.
        """
        config_str = serialize_config_for_logging(self.args, self.cfg)
        self.logger.log_text("experiment/config", config_str, step=0)


    def get_logger(self) -> SpikeLogger:
        """Returns the initialized logger instance."""
        return self.logger

    def get_dirs(self) -> Tuple[str, str]:
        """Returns the log and checkpoint directories."""
        return self.log_dir, self.checkpoint_dir
    
    def log_run_metadata(self):
        """
        Logs high-level metadata about the run to TensorBoard.
        """
        metadata = build_run_metadata(self.cfg)
        lines = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        self.logger.log_text("experiment/run_info", f"```\n{lines}\n```", step=0)
        
        
    def log_model_summary(self, model, input_shape):
        summary_str = str(summary(model, input_size=input_shape, verbose=0))
        # file
        with open(os.path.join(self.exp_dir, "model_summary.txt"), "w") as f:
            f.write(summary_str)
        # tensorboard
        self.logger.log_text("experiment/model_summary", f"```\n{summary_str}\n```", step=0)
        
        
    def log_neuron_counts(self, model, input_shape):
        # get counts
        per_layer, total = count_neurons(model, input_shape)

        # format as multi-line text
        lines = [f"{name}: {cnt}" for name, cnt in per_layer.items()]
        lines.append(f"TOTAL_NEURONS: {total}")
        text = "```\n" + "\n".join(lines) + "\n```"

        # 1) write to disk
        with open(os.path.join(self.exp_dir, "neuron_counts.txt"), "w") as f:
            f.write(text)

        # 2) log to TensorBoard
        self.logger.log_text("experiment/neuron_counts", text, step=0)

        # 3) optionally log each as scalar
        for name, cnt in per_layer.items():
            self.logger.log_scalar(f"neurons/{name}", cnt, step=0)
        self.logger.log_scalar("neurons/total", total, step=0)
        


