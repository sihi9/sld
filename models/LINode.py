import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

class LeakyIntegrator(nn.Module):
    def __init__(self, tau=2.0):
        super().__init__()
        self.neuron = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate.Erf(),  # No spike; just pass membrane potential
            detach_reset=True,  # Membrane does not reset
            v_threshold=1e9,    # Effectively disables spiking
            v_reset=0.0
        )
        self.neuron.store_v_seq = True  # Store membrane potential sequence

    @property
    def v_seq(self):
        return self.neuron.v_seq


    def forward(self, x):
        return self.neuron(x)