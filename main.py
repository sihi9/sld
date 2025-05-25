import argparse
import torch
from torch import optim
from torch.amp import GradScaler


from data.demo_loader import build_demo_dataloader
from models.base_model import SpikingUNetRNN
from engine.trainer import train
from engine.evaluator import evaluate

from utils.visualizations import visualize_predictions
from utils.monitoring import SpikeLogger
from utils.config import load_config


def main():
    args = parse_args()
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
    
    if args.eval_checkpoint:
        print(f"Running evaluation on {args.eval_checkpoint}...")
        model.load_state_dict(torch.load(args.eval_checkpoint, map_location=cfg.train.device, weights_only=True))
        model.to(cfg.train.device)
        model.eval()

        visualize_predictions(model, val_loader, device=cfg.train.device)
        return  # Exit after evaluation
    

    # Optimizer & AMP
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = GradScaler() if cfg.train.amp else None

    # Logger
    logger = SpikeLogger(log_dir=cfg.log.log_dir, 
                         checkpoint_dir=cfg.log.checkpoint_dir,
                         vis_interval=cfg.log.vis_interval)

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
    
    torch.save(model.state_dict(), f"{logger.checkpoint_dir}/checkpoint_final.pth")
    
    logger.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval-checkpoint',
        nargs='?',
        const='checkpoints/checkpoint_final.pth',
        default=None,
        help='Path to checkpoint to evaluate. Use without a value to default to checkpoints/checkpoint_final.pth'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
