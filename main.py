import argparse
import torch
from torch import optim
from torch.amp import GradScaler


from data.loader_utils import DataModule
from models.base_model import SpikingRNN
from models.unet_model import SpikingUNetRNN
from engine.trainer import train
from engine.evaluator import run_final_evaluation_and_save


from utils.visualizations import visualize_predictions
from utils.config import load_config, get_device
from utils.experiment import ExperimentManager


def main():
    args = parse_args()
    cfg = load_config()
    device = get_device()
    print(f"Running on {device} | AMP: {'Enabled' if cfg.train.amp else 'Disabled'}")
        
    exp = ExperimentManager(cfg, args)
    logger = exp.get_logger()
    

    # Data
    loaders = DataModule(cfg).get_loaders()
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    # grab one batch to inspect its spatial size
    x_sample, y_sample = next(iter(train_loader))
    B, T, C_in, H_in, W_in = x_sample.shape
    _, C_out, H_out, W_out = y_sample.shape

    # # Model
    # model = SpikingRNN( 
    #     in_channels=C_in,
    #     out_channels=C_out,        
    #     input_size=(H_in, W_in),
    #     use_recurrent=cfg.model.recurrent,
    #     encoder_channels=cfg.model.encoder_channels,
    #     hidden_dim=cfg.model.hidden_dim,
    #     output_timesteps=cfg.model.output_timesteps,
        
    #     use_plif_encoder=cfg.model.use_plif_encoder,
    #     use_plif_recurrent=cfg.model.use_plif_recurrent,
    #     use_plif_decoder=cfg.model.use_plif_decoder,
    #     init_tau=cfg.model.init_tau
    # )
    model = SpikingUNetRNN(
        in_channels=C_in,
        out_channels=C_out,
        input_size=(H_in, W_in),
        hidden_dim=cfg.model.hidden_dim,
        use_plif_encoder=cfg.model.use_plif_encoder,
        use_plif_recurrent=cfg.model.use_plif_recurrent,
        use_plif_decoder=cfg.model.use_plif_decoder,
        init_tau=cfg.model.init_tau,
        visualize=cfg.log.vis_interval > 0
    )
    
    exp.log_model_summary(model, input_shape=(T, B, C_in, H_in, W_in))
    if cfg.log.vis_interval > 0:    # todo: find a way that doesnt need v_monitor
        exp.log_neuron_counts(model, input_shape=(T, B, C_in, H_in, W_in))
    
    # Load pretrained weights if available
    if args.eval_checkpoint:
        print(f"Running evaluation on {args.eval_checkpoint}...")
        model.load_state_dict(torch.load(args.eval_checkpoint, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        visualize_predictions(model, val_loader, device=cfg.train.device)
        return  # Exit after evaluation
    
    # Optimizer & AMP
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = GradScaler() if cfg.train.amp else None

    # Train
    train(model, train_loader, val_loader,
          optimizer, 
          device=device,
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
        device=device,
        amp=cfg.train.amp,
        epochs=cfg.train.epochs,
        checkpoint_dir=logger.checkpoint_dir
    )
        
    visualize_predictions(model, val_loader, device=device, logger=logger, step=cfg.train.epochs)

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
