import os
import shutil
from datetime import datetime
import yaml

def create_experiment_dir(cfg, args, base_dir="experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg.model.name}_T{cfg.data.time_steps}_lr{cfg.train.lr}_{timestamp}"
    exp_dir = os.path.join(base_dir, exp_name)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    # Save config
    with open(os.path.join(exp_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg.to_dict() if hasattr(cfg, 'to_dict') else cfg, f)

    # Save CLI arguments
    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        f.write(str(args))

    # Metadata
    with open(os.path.join(exp_dir, "run_info.txt"), "w") as f:
        f.write(f"Started: {timestamp}\n")
        f.write(f"Device: {cfg.train.device}\n")
        f.write(f"AMP: {cfg.train.amp}\n")
        f.write(f"Description: {getattr(cfg, 'description', 'N/A')}\n")

    return exp_dir
