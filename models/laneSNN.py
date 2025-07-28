import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional, monitor


class LaneSNN(nn.Module):
    def __init__(
        self,
        input_size=(80, 20),
        hidden_dim=800,
        use_plif=False,
        init_tau=2.0,
        output_timesteps=1,
        visualize=False
    ):
        super().__init__()
        self.input_size = input_size
        self.input_dim = input_size[0] * input_size[1]  # 1600
        self.output_dim = self.input_dim  # 1600
        self.hidden_dim = hidden_dim
        self.use_plif = use_plif
        self.init_tau = init_tau
        self.output_timesteps = output_timesteps
        self.visualize = visualize

        # Layers
        self.input_fc = layer.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.hidden_neuron = self._make_neuron(init_tau, use_plif)
        self.output_fc = layer.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.output_neuron = self._make_neuron(init_tau, use_plif)

        if self.visualize:
            self.monitor = monitor.OutputMonitor(self, (neuron.LIFNode, neuron.ParametricLIFNode))

        functional.set_step_mode(self, step_mode='s')

    def _make_neuron(self, init_tau=2.0, use_plif=False):
        if use_plif:
            return neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan())
        else:
            return neuron.LIFNode(tau=init_tau, surrogate_function=surrogate.ATan())

    def forward(self, x: torch.Tensor, return_logits=True):
        # x: [T, B, 1, H, W]

        T, B, C, H, W = x.shape
        assert (H, W) == self.input_size, f"Expected input shape {(self.input_size)}, got {(H, W)}"

        for m in self.modules():
            if hasattr(m, 'reset'):
                m.reset()

        out = []

        for t in range(T):
            x_t = x[t].view(B, -1)  # [B, 1600]
            h_t = self.input_fc(x_t)
            h_t = self.hidden_neuron(h_t)
            o_t = self.output_fc(h_t)
            o_t = self.output_neuron(o_t)
            out.append(o_t.view(B, 1, H, W))
            

        out = torch.stack(out)  # [T, B, 1, H, W]
        logits = out.mean(dim=0)
        

        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)