import argparse
import os
import torch
from torch import optim
from torch.amp import GradScaler


from data.loader_utils import DataModule
from models.base_model import SpikingRNN
from models.unet_model import SpikingUNetRNN
from engine.trainer import train
from engine.evaluator import run_final_evaluation_and_save


from utils.visualizations import visualize_random_batch, visualize_predictions_video
from utils.config import load_config, get_device
from utils.experiment import ExperimentManager


def main():
    args = parse_args()
    
    resume_path = f"experiments/{args.experiment_name}" if args.experiment_name else None
    cfg = load_config(model=args.model, data=args.data, overrides=args, resume_path=resume_path)
    

    device = get_device()
    print(f"Running on {device} | AMP: {'Enabled' if cfg.train.amp else 'Disabled'}")
    print(f"Using model config: {cfg.model.name}")
    print(f"Using data loader: {cfg.data.loader}")
    
    exp = ExperimentManager(cfg, args)
    logger = exp.get_logger()

    # Data
    loaders = DataModule(cfg).get_loaders()
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test'] 
    
    # grab one batch to inspect its spatial size
    x_sample, y_sample = next(iter(train_loader))
    B, T, C_in, H_in, W_in = x_sample.shape
    _, C_out, H_out, W_out = y_sample.shape

    # # Model
    if cfg.model.name == "base":
        model = SpikingRNN( 
            in_channels=C_in,
            out_channels=C_out,        
            input_size=(H_in, W_in),
            use_recurrent=cfg.model.recurrent,
            encoder_channels=cfg.model.encoder_channels,
            hidden_dim=cfg.model.hidden_dim,
            output_timesteps=cfg.model.output_timesteps,
            
            use_plif_encoder=cfg.model.use_plif_encoder,
            use_plif_recurrent=cfg.model.use_plif_recurrent,
            use_plif_decoder=cfg.model.use_plif_decoder,
            init_tau=cfg.model.init_tau
        )
    elif cfg.model.name == "unet":
        model = SpikingUNetRNN(
            in_channels=C_in,
            out_channels=C_out,
            input_size=(H_in, W_in),
            features=cfg.model.features,
            hidden_dim=cfg.model.hidden_dim,
            use_plif_encoder=cfg.model.use_plif_encoder,
            use_plif_recurrent=cfg.model.use_plif_recurrent,
            use_plif_decoder=cfg.model.use_plif_decoder,
            init_tau=cfg.model.init_tau,
            visualize=cfg.log.vis_interval > 0
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")
    
    exp.log_model_summary(model, input_shape=(T, B, C_in, H_in, W_in))
    # if cfg.log.vis_interval > 0:    # todo: find a way that doesnt need v_monitor
    #     exp.log_neuron_counts(model, input_shape=(T, B, C_in, H_in, W_in))
    
   

    if resume_path:
        checkpoint_filename = {
            "final": "checkpoint_final.pth",
            "best": "checkpoint_best.pth",
            "last": "checkpoint_last.pth"
        }[args.checkpoint_type]

        checkpoint_path = os.path.join(resume_path, "checkpoints", checkpoint_filename)
        
        # Load pretrained weights if available
        print(f"Resuming from experiment: {args.experiment_name}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
      
    

    if args.eval_only:
        print(f"Running evaluation only")
        model.eval()

        visualize_predictions_video(
            model=model,
            dataloader=test_loader,
            device=device,
            save_dir=f"outputs/{args.experiment_name}",
            all_timesteps=False,
            fps=0.5
        )

        #visualize_random_batch(model, val_loader, device=cfg.train.device)
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
        val_loader=test_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        amp=cfg.train.amp,
        epochs=cfg.train.epochs,
        checkpoint_dir=logger.checkpoint_dir
    )
        
    visualize_random_batch(model, test_loader, device=device, logger=logger, step=cfg.train.epochs)

    logger.close()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=["base", "unet"], help='Model profile name')
    parser.add_argument('--data', choices=["demo", "det"], help='Data profile name')

    # CLI overrides
    parser.add_argument('--lr', type=float, dest='train_lr', help='Override training learning rate')
    parser.add_argument('--hidden_dim', type=int, dest='model_hidden_dim', help='Override model hidden dim')

    parser.add_argument('--experiment-name', type=str, help='Name of experiment to resume/evaluate')
    parser.add_argument('--eval-only', action='store_true', help='If set, only run evaluation on given experiment')

    parser.add_argument(
    "--checkpoint-type", type=str, choices=["final", "best", "last"], default="final",
    help="Which checkpoint to evaluate: final (default), best (based on val_iou), or last (latest epoch)"
)
    
    return parser.parse_args()

if __name__ == '__main__':
    main()
