import argparse
import torch
from torch import optim
from torch.amp import GradScaler


from data.demo_loader import build_demo_dataloader
from models.base_model import SpikingUNetRNN
from engine.trainer import train
from engine.evaluator import evaluate

from utils.visualizations import show_sample_triplet
from utils.monitoring import SpikeLogger
from utils.config import load_config


def main():
    cfg = load_config()

    print(f"Running on {cfg.train.device} | AMP: {'Enabled' if cfg.train.amp else 'Disabled'}")

    # Data
    train_loader = build_demo_dataloader(
        batch_size=cfg.data.batch_size,
        time_steps=cfg.data.time_steps,
        input_size=tuple(cfg.data.input_size),
        num_samples=int(cfg.data.num_samples * 0.8)
    )
    val_loader = build_demo_dataloader(
        batch_size=cfg.data.batch_size,
        time_steps=cfg.data.time_steps,
        input_size=tuple(cfg.data.input_size),
        num_samples=int(cfg.data.num_samples * 0.2)
    )

    # Model
    model = SpikingUNetRNN(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        input_size=tuple(cfg.data.input_size),
        encoder_channels=cfg.model.encoder_channels,
        hidden_dim=cfg.model.hidden_dim
    )

    # Optimizer & AMP
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = GradScaler() if cfg.train.amp else None

    # Logger
    logger = SpikeLogger(log_dir=cfg.log.log_dir, vis_interval=cfg.log.vis_interval)

    # Train
    train(model, train_loader, val_loader,
          optimizer, 
          device=cfg.train.device,
          scaler=scaler, 
          epochs=cfg.train.epochs,
          use_amp=cfg.train.amp, 
          logger=logger)

    # Final evaluation
    print("Running final evaluation on validation set...")
    final_loss, final_iou = evaluate(model, val_loader, cfg.train.device, use_amp=cfg.train.amp)
    print(f"Final Loss: {final_loss:.4f}, Final IoU: {final_iou:.4f}")

    logger.close()

if __name__ == '__main__':
    main()
