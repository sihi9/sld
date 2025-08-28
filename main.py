import argparse
import os
import torch
from torch import optim
from torch.amp import GradScaler


from data.loader_utils import DataModule
from models.base_model import SpikingRNN
from models.unet_model_ss import SpikingUNetRNN
from models.laneSNN import LaneSNN
from engine.trainer import train
from engine.evaluator import run_final_evaluation_and_save, evaluate

from utils.monitoring import SpikeLogger
from utils.visualizations import visualize_random_batch, visualize_predictions_video
from utils.config import load_config, get_device
from utils.experiment import ExperimentManager
from utils.memory_profiler import SpikingUNetMemoryAnalyzer

def main():
    args = parse_args()
    
    resume_path = f"experiments/{args.experiment_name}" if args.experiment_name else None
    cfg = load_config(model=args.model, data=args.data, overrides=args, resume_path=resume_path)
    
    # for memory analysis only:
    # cfg.log.vis_interval = -1

    device = get_device()
    print(f"Running on {device} | AMP: {'Enabled' if cfg.train.amp else 'Disabled'}")
    print(f"Using model config: {cfg.model.name}")
    print(f"Using data loader: {cfg.data.loader}")
    print(f"Description: {cfg.description or 'No description provided'}")
    
    exp = ExperimentManager(cfg, args)
    logger : SpikeLogger = exp.get_logger()

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
            fc_bottleneck=cfg.model.fc_bottleneck,
            fc_recurrent=cfg.model.fc_recurrent,
            conv_recurrent=cfg.model.conv_recurrent,
            hidden_dim=cfg.model.hidden_dim,
            output_timesteps=cfg.model.output_timesteps,
            soft_reset=cfg.model.soft_reset,
            skip_connections=cfg.model.skip_connections,
            analog=cfg.model.analog,  # Use analog skips if specified
            initial_scaling=cfg.model.initial_scaling,
            use_plif_encoder=cfg.model.use_plif_encoder,
            use_plif_recurrent=cfg.model.use_plif_recurrent,
            use_plif_decoder=cfg.model.use_plif_decoder,
            init_tau_recurrent=cfg.model.init_tau_recurrent,
            init_tau_encoder=cfg.model.init_tau_encoder,
            init_tau_decoder=cfg.model.init_tau_decoder,
            visualize=cfg.log.vis_interval > 0
        )
    elif cfg.model.name == "lanesnn":
        model = LaneSNN(
            input_size=(H_in, W_in),
            hidden_dim=cfg.model.hidden_dim,
            use_plif=cfg.model.use_plif,
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
            "latest": "checkpoint_latest.pth"
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
        
        final_loss, final_iou = evaluate(model, test_loader, device, use_amp=False)
        logger.log_scalar("test/final_IoU", final_iou, step=0)
        logger.log_scalar("test/final_loss", final_loss, step=0)
        print(f"Final evaluation loss: {final_loss:.4f}, IoU: {final_iou:.4f}")
    
        #memory_analysis(model, input_shape=(C_in, H_in, W_in), timesteps=T, batch_size=cfg.data.batch_size)
        visualize_random_batch(model, test_loader, device=device, n=cfg.data.batch_size, logger=logger, step=cfg.train.epochs)
        visualize_predictions_video(
            model=model,
            dataloader=val_loader,
            device=device,
            save_dir=f"outputs/{args.experiment_name}",
            all_timesteps=False,
            fps=2
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
        logger=logger
    )
    
    report = memory_analysis(model, input_shape=(C_in, H_in, W_in), timesteps=T, batch_size=cfg.data.batch_size)
    logger.log_text("memory_report", str(report))
        
    visualize_random_batch(model, test_loader, device=device, n=cfg.data.batch_size, logger=logger, step=cfg.train.epochs)
    logger.close()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=["base", "unet", "lanesnn"], help='Model profile name')
    parser.add_argument('--data', choices=["demo", "det", "carla"], help='Data profile name')

    # CLI overrides
    parser.add_argument('--lr', type=float, dest='train_lr', help='Override training learning rate')
    parser.add_argument('--hidden-dim', type=int, dest='model_hidden_dim', help='Override model hidden dim')
    parser.add_argument('--epochs', type=int, dest='train_epochs', default=None, help='Number of training epochs')

    parser.add_argument('--features', nargs='+', type=int, dest='model_features', help='Override U-Net features')
    parser.add_argument('--fc-bottleneck', dest='model_fc_bottleneck', action='store_true', help='Use FC bottleneck')
    parser.add_argument('--no-fc-bottleneck', dest='model_fc_bottleneck', action='store_false', help='Do not use FC bottleneck')
    parser.set_defaults(model_fc_bottleneck=None)
    
    parser.add_argument('--hard-reset', dest='model_soft_reset', action='store_false', help='Use hard reset')
    parser.add_argument('--soft-reset', dest='model_soft_reset', action='store_true', help='Use soft reset')
    parser.set_defaults(model_soft_reset=None)
    parser.add_argument('--no-skip-connections', dest='model_skip_connections', action='store_false', help='Do not use skip connections')
    parser.set_defaults(model_skip_connections=None)
    
    parser.add_argument('--analog', dest='model_analog', action='store_true', help='Use analog skips. Note that this might not work without initial scaling block, as initial neurons would not spike')
    parser.add_argument('--not-analog', dest='model_analog', action='store_false', help='Do not use analog skips')
    parser.set_defaults(model_analog=None)
    
    parser.add_argument('--fc-recurrent', dest='model_fc_recurrent', action='store_true', help='Use recurrent bottleneck')
    parser.add_argument('--no-fc-recurrent', dest='model_fc_recurrent', action='store_false', help='Do not use recurrent bottleneck')
    parser.set_defaults(model_fc_recurrent=None)
    
    parser.add_argument('--conv-recurrent', dest='model_conv_recurrent', action='store_true', help='Use recurrent encoder')
    parser.add_argument('--no-conv-recurrent', dest='model_conv_recurrent', action='store_false', help='Do not use recurrent encoder')
    parser.set_defaults(model_conv_recurrent=None)
    
    parser.add_argument('--static-data', dest='data_use_static', action='store_true', help='Use static data loader')
    parser.add_argument('--used-T', dest='data_used_T', type=int, default=None, help='Override data timesteps')
    parser.add_argument('--initial-scaling', dest='model_initial_scaling', type=int, default=None, help='Initial scaling factor for input size, used to calculate downscaling factor')
    parser.add_argument('--downscale', dest='data_downscale', default=None, type=int, help='Downscaling factor for input size')
    
    parser.add_argument('--description', type=str, dest='cfg_description', help='Override config description')
    
    parser.add_argument('--experiment-name', type=str, help='Name of experiment to resume/evaluate')
    parser.add_argument('--eval-only', action='store_true', help='If set, only run evaluation on given experiment')

    parser.add_argument(
    "--checkpoint-type", type=str, choices=["final", "best", "latest"], default="final",
    help="Which checkpoint to evaluate: final (default), best (based on val_iou), or last (latest epoch)"
    )
    
    return parser.parse_args()

def memory_analysis(model, input_shape, timesteps, batch_size):
    analyzer = SpikingUNetMemoryAnalyzer(model, 'cuda')
    model.visualize = False
    model.output_monitor.remove_hooks()  # Disable output monitor for memory analysis
    print(f"output monitors: {model.output_monitor}")
    report = analyzer.generate_memory_report(input_shape=input_shape,
                                             timesteps_range=[timesteps],
                                             batch_sizes=[batch_size])
    return report
    # result_train = analyzer.profile_inference_sequence(
    #     input_shape=input_shape,
    #     timesteps=timesteps,
    #     batch_size=batch_size,
    #     with_gradients=True 
    # )
    # #print(f"Memory usage for training sequence: {result_train}")
    
    # result_test = analyzer.profile_inference_sequence(
    #     input_shape=input_shape,
    #     timesteps=timesteps,
    #     batch_size=batch_size,
    #     with_gradients=False  # For inference
    # )
    #print(f"Memory usage for inference sequence: {result_test}")
    
    
if __name__ == '__main__':
    main()
