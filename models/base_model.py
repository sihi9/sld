import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional, monitor
from .LINode import LeakyIntegrator


class SpikingUNetRNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 recurrent=True,
                 input_size=(128, 128), 
                 hidden_dim=4096, # currently unused
                 encoder_channels=(2, 4),
                 output_timesteps=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.recurrent = recurrent
        self.input_size = input_size  # (H, W)
        self.hidden_dim = hidden_dim
        self.encoder_channels = encoder_channels
        self.output_timesteps = output_timesteps
      
        # Output scaling and bias parameters
        self.output_scale = nn.Parameter(torch.tensor(5.0))
        self.output_bias = nn.Parameter(torch.tensor(0.5))


        h, w = input_size
        assert h % 4 == 0 and w % 4 == 0, "Input size must be divisible by 4 for 2x stride-2 downsamples."

        # --- Encoder ---
        c1, c2 = encoder_channels
        self.encoder = nn.Sequential(
            layer.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),  # H/2 x W/2
            layer.BatchNorm2d(c1),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),           # H/4 x W/4
            layer.BatchNorm2d(c2),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

        self.h_down = h // 4
        self.w_down = w // 4
        self.flat_dim = c2 * self.h_down * self.w_down

        # --- Recurrent bottleneck ---
        self.flatten = nn.Flatten(start_dim=2)
        self.recurrent = layer.LinearRecurrentContainer(
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=0.5, detach_reset=False),
            in_features=self.flat_dim,
            out_features=self.flat_dim,
            bias=True
        )
        self.feedforward_bottleneck = layer.Linear(self.flat_dim, self.flat_dim)


        # --- Decoder ---
        self.linear_decoder = layer.Linear(self.flat_dim, self.flat_dim, bias=False)
        self.decoder_neuron = neuron.LIFNode(surrogate_function=surrogate.ATan())

        self.decoder_conv = nn.Sequential(
            layer.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1, bias=False),  # H/2 x W/2
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.ConvTranspose2d(c1, out_channels, kernel_size=4, stride=2, padding=1, bias=False),  # H x W
            LeakyIntegrator()
        )

        functional.set_step_mode(self, step_mode='m')
        
        # size can be reduced. See: https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/monitor.html
        self.output_monitor = monitor.OutputMonitor(self, neuron.LIFNode)
        self.v_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=self, instance=neuron.LIFNode)

       # Enable voltage recording
        for m in self.modules():
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = True

        with torch.no_grad():
            self.encoder[0].weight += 0.5 
            self.encoder[3].weight += 0.5 
            
            self.recurrent.rc.weight *= 5
            self.linear_decoder.weight += 0.5
            
            self.decoder_conv[0].weight += 0.1
            self.decoder_conv[2].weight += 0.1


    def forward(self, x: torch.Tensor, return_seq=False):
        # x: [T, B, in_channels, H, W]
        x = self.encoder(x)                            # [T, B, c2, H/4, W/4]
        x_flat = self.flatten(x)      
         
        # [T, B, flat_dim]
        if self.recurrent:
            x_latent = self.recurrent(x_flat)
        else:
            x_latent = self.feedforward_bottleneck(x_flat)  

        x_decoded = self.linear_decoder(x_latent)      # [T, B, flat_dim]
        x_decoded = self.decoder_neuron(x_decoded)
        x_reshaped = x_decoded.view(-1, x.shape[1], self.encoder_channels[1],
                                    self.h_down, self.w_down)  # [T, B, c2, H/4, W/4]

        x_out = self.decoder_conv(x_reshaped)          # [T, B, out_channels, H, W]
        
        # --- Use membrane potential instead of spike output ---
        v_seq = self.decoder_conv[-1].v_seq  # [T, B, out_channels, H, W]
    
            
        if return_seq: # todo: this does not work yet
            return v_seq
        else:
            v_agg = self.aggregate_output(v_seq)        # [B, 1, H, W]
            logits = self.output_scale * (v_agg - self.output_bias)
            probabilities = torch.sigmoid(logits)
            return probabilities
        
    def aggregate_output(self, v_seq: torch.Tensor):
        """
        Aggregate the output from the sequence of membrane potentials.
        v_seq: [T, B, 1, H, W]
        returns: [B, 1, H, W]
        """
        if self.output_timesteps == 1:
            v_out = v_seq[-1]
        elif self.output_timesteps > 1:
            v_out = v_seq[-self.output_timesteps:].mean(dim=0)
        else:  # e.g., -1 means use all time steps
            v_out = v_seq.mean(dim=0)
        return v_out

