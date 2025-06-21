import torch
import os
from torch import prod
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from spikingjelly import visualizing

from spikingjelly.activation_based import functional

import matplotlib
import matplotlib.pyplot as plt



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

    #print("Logging spikes...")
    #log_spikes_from_monitor(model, logger, epoch)

    #print("Logging membrane potentials...")
    #log_membrane_from_monitor(model, logger, epoch)
    
    print("Logging spike rates...")
    log_spike_rate_summary(model, logger, epoch)


def log_spikes_from_monitor(model, logger: SpikeLogger, epoch: int):
    """
    Logs spike activity and membrane potentials from SpikingJelly monitors.
    Only logs spikes for batch index 0 and separates by channel if present.
    """
    for layer_name in model.output_monitor.monitored_layers:
        if layer_name == "recurrent.sub_module":
            print("recurrent")
            
        print(f"Logging spikes for layer: {layer_name}")
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
            fig = plot_1d_spikes(
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
                fig = plot_1d_spikes(
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



def log_spike_rate_summary(model, logger: SpikeLogger, epoch: int):
    for layer_name in model.output_monitor.monitored_layers:
        records = model.output_monitor[layer_name]
        if not records or len(records) == 0:
            continue
        
        spikes = records[0].detach() # [T, B, ...]   
        spikes = spikes.float().mean(dim=1)   # average over batch
        
        if spikes.dim() == 2:
            rate = spikes.float().mean(dim=0).cpu().numpy()  # [N]
        elif spikes.dim() == 4:
            rate = spikes.float().mean(dim=(0, 2, 3)).cpu().numpy()  # [C]
        else:
            print(f"⚠️ Skipped {layer_name}: unexpected shape {records[0].shape}")

        fig, ax = plt.subplots()
        ax.bar(np.arange(len(rate)), rate)
        ax.set_title(f"Avg Spike Rate - {layer_name}")
        ax.set_xlabel("Neuron Index" if rate.ndim == 1 else "Channel Index")
        ax.set_ylabel("Firing Rate")
        logger.writer.add_figure(f"spike_rate_summary/{layer_name}", fig, epoch)
        

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




def plot_1d_spikes(spikes: np.asarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                   plot_firing_rate=True, firing_rate_map_title='firing rate', figsize=(12, 8), dpi=200):
    '''


    :param spikes: shape=[T, N]的np数组，其中的元素只为0或1，表示N个时长为T的脉冲数据
    :param title: 热力图的标题
    :param xlabel: 热力图的x轴的label
    :param ylabel: 热力图的y轴的label
    :param int_x_ticks: x轴上是否只显示整数刻度
    :param int_y_ticks: y轴上是否只显示整数刻度
    :param plot_firing_rate: 是否画出各个脉冲发放频率
    :param firing_rate_map_title: 脉冲频率发放图的标题
    :param dpi: 绘图的dpi
    :return: 绘制好的figure

    画出N个时长为T的脉冲数据。可以用来画N个神经元在T个时刻的脉冲发放情况，示例代码：

    .. code-block:: python

        import torch
        from spikingjelly.activation_based import neuron
        from spikingjelly import visualizing
        from matplotlib import pyplot as plt
        import numpy as np

        lif = neuron.LIFNode(tau=100.)
        x = torch.rand(size=[32]) * 4
        T = 50
        s_list = []
        v_list = []
        for t in range(T):
            s_list.append(lif(x).unsqueeze(0))
            v_list.append(lif.v.unsqueeze(0))

        s_list = torch.cat(s_list)
        v_list = torch.cat(v_list)

        visualizing.plot_1d_spikes(spikes=np.asarray(s_list), title='Membrane Potentials', xlabel='Simulating Step',
                                   ylabel='Neuron Index', dpi=200)
        plt.show()

    .. image:: ./_static/API/visualizing/plot_1d_spikes.*
        :width: 100%

    '''
    if spikes.ndim != 2:
        raise ValueError(f"Expected 2D array, got {spikes.ndim}D array instead")

    spikes_T = spikes.T
    if plot_firing_rate:
        fig = plt.figure(tight_layout=True, figsize=figsize, dpi=dpi)
        gs = matplotlib.gridspec.GridSpec(1, 5)
        spikes_map = fig.add_subplot(gs[0, 0:4])
        firing_rate_map = fig.add_subplot(gs[0, 4])
    else:
        fig, spikes_map = plt.subplots()

    spikes_map.set_title(title)
    spikes_map.set_xlabel(xlabel)
    spikes_map.set_ylabel(ylabel)

    spikes_map.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    spikes_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))

    spikes_map.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    spikes_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    spikes_map.set_xlim(-0.5, spikes_T.shape[1] - 0.5)
    spikes_map.set_ylim(-0.5, spikes_T.shape[0] - 0.5)
    spikes_map.invert_yaxis()
    N = spikes_T.shape[0]
    T = spikes_T.shape[1]
    t = np.arange(0, T)
    t_spike = spikes_T * t
    mask = (spikes_T == 1)  # eventplot中的数值是时间发生的时刻，因此需要用mask筛选出

    colormap = plt.get_cmap('tab10')  # cmap的种类参见https://matplotlib.org/gallery/color/colormap_reference.html

    positions = [t_spike[i][mask[i]] for i in range(N)]
    colors = [colormap(i % 10) for i in range(N)]
    spikes_map.eventplot(positions, lineoffsets=np.arange(N), colors=colors)

    if plot_firing_rate:
        firing_rate = np.mean(spikes_T, axis=1, keepdims=True)

        max_rate = firing_rate.max()
        min_rate = firing_rate.min()

        firing_rate_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        firing_rate_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        firing_rate_map.imshow(firing_rate, cmap='magma', aspect='auto')
        for i in range(firing_rate.shape[0]):
            firing_rate_map.text(0, i, f'{firing_rate[i][0]:.2f}', ha='center', va='center', color='w' if firing_rate[i][0] < 0.7 * max_rate or min_rate == max_rate else 'black')
        firing_rate_map.get_xaxis().set_visible(False)
        firing_rate_map.set_title(firing_rate_map_title)
    return fig

