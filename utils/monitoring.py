import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from spikingjelly import visualizing


class SpikeLogger:
    def __init__(self, 
                 log_dir="./runs", 
                 vis_interval=1,
                 checkpoint_dir="./checkpoints"):
        self.writer = SummaryWriter(log_dir)
        self.vis_interval = vis_interval
        self.checkpoint_dir = checkpoint_dir

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()

    def log_text(self, tag, text_string, step=0):
        self.writer.add_text(tag, text_string, step)


def log_from_monitors(model, logger: SpikeLogger, epoch: int):
    """
    Logs spike activity and membrane potentials from SpikingJelly monitors.
    Only logs spikes for batch index 0 and separates by channel if present.
    """
    if logger.vis_interval is None or logger.vis_interval <= 0:
        return
    if (epoch + 1) % logger.vis_interval != 0:
        return

    print("Logging spikes...")
    log_spikes_from_monitor(model, logger, epoch)

    print("Logging membrane potentials...")
    log_membrane_from_monitor(model, logger, epoch)


def log_spikes_from_monitor(model, logger: SpikeLogger, epoch: int):
    """
    Logs spike activity and membrane potentials from SpikingJelly monitors.
    Only logs spikes for batch index 0 and separates by channel if present.
    """
    for layer_name in model.output_monitor.monitored_layers:
        records = model.output_monitor[layer_name]

        if not records or len(records) == 0:
            continue

        spikes = records[0].detach()  # [T, B, ...]
        #print(f"Layer: {layer_name}, Records: {spikes.shape}")

        # Pick only batch 0
        spikes = spikes[:, 0]  # [T, ...]
        
        if spikes.dim() == 2:
            # [T, N] already (e.g., Linear layers)
            spikes_flat = spikes.cpu().numpy()
            fig = visualizing.plot_1d_spikes(
                spikes=spikes_flat,
                title=f"Spikes - {layer_name}",
                xlabel="Time step",
                ylabel="Neuron index",
                dpi=150
            )
            logger.writer.add_figure(f"spikes/{layer_name}", fig, epoch)

        elif spikes.dim() == 4:
            # [T, C, H, W] → plot each channel separately
            T, C, H, W = spikes.shape
            for c in range(C):
                spikes_c = spikes[:, c]  # [T, H, W]
                spikes_flat = spikes_c.view(T, -1).cpu().numpy()  # [T, H*W]
                fig = visualizing.plot_1d_spikes(
                    spikes=spikes_flat,
                    title=f"Spikes - {layer_name}/ch{c}",
                    xlabel="Time step",
                    ylabel="Neuron index",
                    dpi=150
                )
                logger.writer.add_figure(f"spikes/{layer_name}/ch{c}", fig, epoch)

        else:
            print(f"⚠️ Skipped {layer_name}: unexpected shape {spikes.shape}")
            
    
def log_membrane_from_monitor(model, logger: SpikeLogger, epoch: int):
    """
    Logs membrane potentials from SpikingJelly monitors.
    Only logs spikes for batch index 0 and separates by channel if present.
    """
    for layer_name in model.v_monitor.monitored_layers:
        records = model.v_monitor[layer_name]

        if not records or len(records) == 0 or records[0] is None:
            continue
        
        v_seq = records[0].detach()
      
        #print(f"Layer: {layer_name}, v_seq shape: {v_seq.shape}")

        # Select batch 0
        v_seq = v_seq[:, 0]  # [T, ...]

        if v_seq.dim() == 2:
            # [T, N]
            v_flat = v_seq.cpu().numpy()
            fig = visualizing.plot_2d_heatmap(
                array=v_flat,
                title=f"Membrane V[t] - {layer_name}",
                xlabel='Time step',
                ylabel='Neuron',
                dpi=150
            )
            logger.writer.add_figure(f"membrane/{layer_name}", fig, epoch)

        elif v_seq.dim() == 4:
            # [T, C, H, W]
            T, C, H, W = v_seq.shape
            for c in range(C):
                v_ch = v_seq[:, c]  # [T, H, W]
                v_flat = v_ch.view(T, -1).cpu().numpy()
                fig = visualizing.plot_2d_heatmap(
                    array=v_flat,
                    title=f"Membrane V[t] - {layer_name}/ch{c}",
                    xlabel='Time step',
                    ylabel='Neuron index',
                    dpi=150
                )
                logger.writer.add_figure(f"membrane/{layer_name}/ch{c}", fig, epoch)

        else:
            print(f"⚠️ Skipped {layer_name}: unexpected shape {v_seq.shape}")
