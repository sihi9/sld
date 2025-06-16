import torch
import os
from torch import prod
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from spikingjelly import visualizing

from spikingjelly.activation_based import functional




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
        if layer_name == "recurrent.sub_module":
            print("recurrent")
            
        records = model.output_monitor[layer_name]

        if not records or len(records) == 0:
            continue

        # select the first batch
        spikes = records[0].detach()  # [T, B, ...]

        # Pick first sample in batch
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
        
        # Select batch 0
        v_seq = records[0].detach()
      
        #print(f"Layer: {layer_name}, v_seq shape: {v_seq.shape}")

        # Pick first sample in batch
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





def count_neurons(model, input_shape):
    """
    Runs one forward pass of `model` on a dummy input of shape `input_shape`
    (T, B, C, H, W) with B=1, then inspects model.v_monitor to get v_seq shapes
    for each spiking‐neuron layer. Returns:
      - per_layer: dict[layer_name] = num_neurons
      - total: sum of all num_neurons
    """
    # 1) clear any existing state
    functional.reset_net(model)
    # 2) make a dummy batch (all zeros) on same device
    T, B, C, H, W = input_shape
    device = next(model.parameters()).device
    dummy = torch.zeros(T, B, C, H, W, device=device)
    # 3) forward (we only need v_seq monitors; outputs can be discarded)
    _ = model(dummy, return_seq=True)
    # 4) iterate over each monitored module
    per_layer = {}
    total = 0
    for layer_name in model.v_monitor.monitored_layers:
        # model.v_monitor[layer_name] is a list of v_seq tensors, one per forward
        records = model.v_monitor[layer_name]
        if not records or records[0] is None:
            continue
        v_seq = records[0]    # shape [T, B, …]
        # count a single time‐slice and batch‐slice
        shape = v_seq.shape[2:]
        n_neurons = int(prod(torch.tensor(shape)))
        per_layer[layer_name] = n_neurons
        total += n_neurons
    return per_layer, total

