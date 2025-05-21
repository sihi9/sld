import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from spikingjelly import visualizing


class SpikeLogger:
    def __init__(self, log_dir="./runs", vis_interval=1):
        self.writer = SummaryWriter(log_dir)
        self.vis_interval = vis_interval

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()


def log_from_monitors(model, logger: SpikeLogger, epoch: int):
    """
    Logs spike activity and membrane potentials from SpikingJelly monitors.
    """
    if logger.vis_interval  is None or logger.vis_interval  <= 0:
        return
    if epoch % logger.vis_interval != 0:
        return
    print("Logging spikes...")
    # Log spikes
    for layer_name in model.output_monitor.monitored_layers:
        records = model.output_monitor[layer_name]
        if not records:
            continue
        spikes = records[0].detach()  # [T, B, N]
        
        spikes_flat = spikes.view(spikes.shape[0], -1).cpu().numpy()  # [T, N]

        fig = visualizing.plot_1d_spikes(
            spikes=spikes_flat,
            title=f"Spikes - {layer_name}",
            xlabel="Time step",
            ylabel="Neuron index",
            dpi=150
        )
        logger.writer.add_figure(f"spikes/{layer_name}", fig, epoch)

        

    print("Logging membrane potentials...")
    # Log membrane potential heatmap
    for layer_name in model.v_monitor.monitored_layers:
        records = model.v_monitor[layer_name]
        for v_seq in records:
            if v_seq is None:
                continue
            v_seq = v_seq.detach()  # [T, B, N]
            v_flat = v_seq.view(v_seq.shape[0], -1).cpu()
            fig = visualizing.plot_2d_heatmap(
                array=v_flat.numpy(),
                title=f"Membrane V[t] - {layer_name}",
                xlabel='Time step',
                ylabel='Neuron',
                dpi=150
            )
            logger.writer.add_figure(f"membrane/{layer_name}", fig, epoch)

    # Clear for next epoch
    print("Clearing recorded data...")
    model.output_monitor.clear_recorded_data()
    model.v_monitor.clear_recorded_data()
