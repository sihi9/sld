import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import neuron, surrogate

class PLIFNode(neuron.BaseNode):
    """
    Parametric Leaky Integrate-and-Fire (PLIF) neuron with a learnable membrane time constant.

    The membrane potential update is:
        V[t] = V[t-1] + alpha * (X[t] - (V[t-1] - V_reset))   [hard reset]
        V[t] = V[t-1] + alpha * (X[t] - V[t-1])               [soft reset]
    where alpha = sigmoid(w) and tau = 1 / alpha
    """

    def __init__(self,
                 init_tau=2.0,
                 v_threshold=1.0,
                 v_reset=0.0,
                 surrogate_function=surrogate.ATan(),
                 detach_reset=True):
        """
        :param init_tau: initial membrane time constant τ (> 1.0)
        :param v_threshold: spike threshold
        :param v_reset: reset voltage after spike (set to None for soft reset)
        :param surrogate_function: surrogate gradient function
        :param detach_reset: whether to detach spike from graph in reset
        """
        super().__init__(v_threshold=v_threshold,
                         v_reset=v_reset,
                         surrogate_function=surrogate_function,
                         detach_reset=detach_reset)

        self.init_tau = init_tau
        self.init_w = -math.log(init_tau - 1.0)
        
        # Start with scalar; shape will be adjusted on first use
        self.w = nn.Parameter(torch.tensor(self.init_w).view(1), requires_grad=True)
        self._w_broadcasted = False

    def neuronal_charge(self, x: torch.Tensor):
        """
        Update membrane potential using PLIF dynamics.
        """
        if not self._w_broadcasted:
            expanded_shape = x.shape[1:]  # exclude batch/time dim
            if expanded_shape != self.w.shape:
                new_w = torch.full(expanded_shape, self.init_w, dtype=x.dtype, device=x.device)
                self.w = nn.Parameter(new_w, requires_grad=True)
            self._w_broadcasted = True
        
        alpha = self.w.sigmoid()
        if self.v_reset is None:
            self.v = self.v + (x - self.v) * alpha
        else:
            self.v = self.v + (x - (self.v - self.v_reset)) * alpha

    def tau(self) -> float:
        """
        Get the current membrane time constant τ.
        """
        return 1.0 / self.w.sigmoid().item()

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau():.4f}'
