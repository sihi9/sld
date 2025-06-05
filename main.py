import argparse
import torch
import os
import yaml
from torch import optim
from torch.amp import GradScaler


from data.demo_loader import build_demo_dataloader
from data.loader_utils import DataModule
from models.base_model import SpikingUNetRNN
from engine.trainer import train
from engine.evaluator import run_final_evaluation_and_save


from utils.visualizations import visualize_predictions
from utils.monitoring import SpikeLogger
from utils.config import load_config
from utils.experiment import ExperimentManager


def main():
    args = parse_args()
    cfg = load_config()

    print(f"Running on {cfg.train.device} | AMP: {'Enabled' if cfg.train.amp else 'Disabled'}")
        
    exp = ExperimentManager(cfg, args)
    logger = exp.get_logger()
    

    # Data
    loaders = DataModule(cfg).get_loaders()
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    # Model
    model = SpikingUNetRNN(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        recurrent=cfg.model.recurrent,
        input_size=tuple(cfg.data.input_size),
        encoder_channels=cfg.model.encoder_channels,
        hidden_dim=cfg.model.hidden_dim,
        output_timesteps=cfg.model.output_timesteps,
    )
    
    
    # Load pretrained weights if available
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

    # Train
    train(model, train_loader, val_loader,
          optimizer, 
          device=cfg.train.device,
          scaler=scaler, 
          epochs=cfg.train.epochs,
          use_amp=cfg.train.amp, 
          logger=logger,
          save_intermediate=cfg.train.save_intermediate,)


    # Final evaluation
    run_final_evaluation_and_save(
        model=model,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        cfg=cfg,
        checkpoint_dir=logger.checkpoint_dir
    )
        
    visualize_predictions(model, val_loader, device=cfg.train.device, logger=logger, step=cfg.train.epochs)

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
