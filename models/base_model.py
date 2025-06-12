import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional, monitor
from .LINode import LeakyIntegrator
from .PLIFNode import PLIFNode


class SpikingUNetRNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                use_recurrent=True,
                input_size=(128, 128), 
                hidden_dim=4096,
                encoder_channels=(2, 4),
                output_timesteps=1,
                use_plif_encoder=False,
                use_plif_recurrent=False,
                use_plif_decoder=False,
                init_tau=2.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_recurrent = use_recurrent
        self.input_size = input_size  # (H, W)
        self.hidden_dim = hidden_dim
        self.encoder_channels = encoder_channels
        self.output_timesteps = output_timesteps
        self.use_plif_encoder = use_plif_encoder
        self.use_plif_recurrent = use_plif_recurrent
        self.use_plif_decoder = use_plif_decoder
        self.init_tau = init_tau
        
        # Output scaling and bias parameters
        self.output_scale = nn.Parameter(torch.tensor(5.0))
        self.output_bias = nn.Parameter(torch.tensor(0.5))


        h, w = input_size
        assert h % 4 == 0 and w % 4 == 0, "Input size must be divisible by 4 for 2x stride-2 downsamples."

        # --- Encoder ---
        c1, c2 = encoder_channels
        self.encoder = nn.Sequential(
            layer.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            layer.BatchNorm2d(c1),
            self._make_neuron(use_plif=self.use_plif_encoder),
            layer.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            layer.BatchNorm2d(c2),
            self._make_neuron(use_plif=self.use_plif_encoder)
        )

        self.h_down = h // 4
        self.w_down = w // 4
        self.flat_dim = c2 * self.h_down * self.w_down

        # --- Recurrent bottleneck ---
        self.flatten = nn.Flatten(start_dim=2)
        
        
        self.recurrent = layer.LinearRecurrentContainer(
            self._make_neuron(use_plif=self.use_plif_recurrent),
            in_features=self.flat_dim,
            out_features=self.flat_dim,
            bias=True
        )
                
        self.feedforward_bottleneck = layer.Linear(self.flat_dim, self.flat_dim)


        # --- Decoder ---
        self.linear_decoder = layer.Linear(self.flat_dim, self.flat_dim, bias=False)
        self.decoder_neuron = self._make_neuron(use_plif=self.use_plif_decoder)

        self.decoder_conv = nn.Sequential(
            layer.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1, bias=False),
            self._make_neuron(use_plif=self.use_plif_decoder),
            layer.ConvTranspose2d(c1, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            LeakyIntegrator()
        )
        
        functional.set_step_mode(self.recurrent, step_mode='m')
        functional.set_step_mode(self, step_mode='m')
        
        # size can be reduced. See: https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/monitor.html
        self.output_monitor = monitor.OutputMonitor(self, (neuron.LIFNode, neuron.ParametricLIFNode))
        self.v_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=self, instance=(neuron.LIFNode, neuron.ParametricLIFNode))


        for m in self.modules():
            if isinstance(m, (neuron.LIFNode, neuron.ParametricLIFNode)):
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
        if self.use_recurrent:
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

    def _make_neuron(self, use_plif=False):
        if use_plif:
            return neuron.ParametricLIFNode(init_tau=self.init_tau, surrogate_function=surrogate.ATan()) 
        else:
            return neuron.LIFNode(surrogate_function=surrogate.ATan())

