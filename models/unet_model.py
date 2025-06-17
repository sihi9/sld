import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional, monitor
from .LINode import LeakyIntegrator


class SpikingUNetRNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_size=(128, 128),
        features=(64, 128, 256, 512),
        hidden_dim=512,
        output_timesteps=1,
        use_plif_encoder=False,
        use_plif_recurrent=False,
        use_plif_decoder=False,
        init_tau=2.0,
        visualize=False
    ):
        super().__init__()
        self.features = features
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_timesteps = output_timesteps
        self.use_plif_encoder = use_plif_encoder
        self.use_plif_recurrent = use_plif_recurrent
        self.use_plif_decoder = use_plif_decoder
        self.init_tau = init_tau
        self.visualize = visualize

        
        # Output scaling and bias parameters
        self.output_scale = nn.Parameter(torch.tensor(5.0))
        self.output_bias = nn.Parameter(torch.tensor(0.5))
        
        
        H, W = input_size
        
        depth = len(features)  # Number of downsampling layers
        downscaling_factor = 2 ** depth
        assert (
            H % downscaling_factor == 0 
            and W % downscaling_factor == 0
        ), f"Input size must be divisible by {downscaling_factor} for {len(features)}x stride-2 downsamples."


        bottom_H = H // (2 ** depth)
        bottom_W = W // (2 ** depth)
        
        print(f"Input size: {input_size}, downscaling factor: {downscaling_factor}")
       


        # Encoder path
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(self.double_conv(prev_channels, feat, use_plif_encoder))
            prev_channels = feat

        # Bottom pooling
        self.pool = layer.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck recurrent fully-connected
        flat_dim = features[-1] * bottom_H * bottom_W
        
        self.reduce_fc = layer.Linear(
            in_features=flat_dim,
            out_features=hidden_dim,
            bias=False,
            step_mode='m'
        )
                
        self.recurrent = layer.LinearRecurrentContainer(
            self._make_neuron(use_plif=self.use_plif_recurrent),
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=True
        )
        
        self.expand_fc = layer.Linear(
            in_features=hidden_dim,
            out_features=flat_dim,
            bias=False,
            step_mode='m'
        )

        # Decoder path
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_ch = features[-1]
        for feat, skip_ch in zip(reversed(features[:-1]), reversed(features[1:])):
            # up from prev_ch to feat
            self.upconvs.append(
                layer.ConvTranspose2d(prev_ch, feat, kernel_size=2, stride=2)
            )
            # decoder expects skip_ch + feat channels
            self.decoders.append(
                self.double_conv(skip_ch + feat, feat, use_plif_decoder)
            )
            prev_ch = feat

        self.upconvs.append(
            layer.ConvTranspose2d(prev_ch, features[0], kernel_size=2, stride=2)
        )
        self.decoders.append(
            self.double_conv(features[0], features[0], use_plif_decoder)
        )

        # Final 1x1 conv
        self.final_conv = layer.Conv2d(features[0], out_channels, kernel_size=1)
        self.output_integrator = LeakyIntegrator()
        
        

        if self.visualize:
            self.output_monitor = monitor.OutputMonitor(self, 
                                                        (neuron.LIFNode, neuron.ParametricLIFNode, layer.LinearRecurrentContainer))
            self.v_monitor = monitor.AttributeMonitor('v_seq', 
                                                    pre_forward=False, 
                                                    net=self, 
                                                    instance=(neuron.LIFNode, neuron.ParametricLIFNode))
            
            for m in self.modules():
                if isinstance(m, (neuron.LIFNode, neuron.ParametricLIFNode)):
                    m.store_v_seq = True
                    
        
        # Use multi-step mode
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor, return_seq : bool = False) -> torch.Tensor:
        # x: [T, B, C, H, W]
        # Reset states
        for m in self.modules():
            if hasattr(m, 'reset'):
                m.reset()

        skips = []
        # Encoder
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        T, B, C, Hb, Wb = x.shape
        # Flatten spatial dims
        x_flat = x.view(T, B, -1)
        # reduce
        reduced = self.reduce_fc(x_flat)
        # recurrent
        h_rec = self.recurrent(reduced)
        # expand
        x_exp = self.expand_fc(h_rec)
        x = x_exp.view(T, B, C, Hb, Wb)

        # Decoder
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            if i < len(self.upconvs) - 1:
                skip = skips.pop()
                if x.shape[-2:] != skip.shape[-2:]:
                    dy = skip.size(-2) - x.size(-2)
                    dx = skip.size(-1) - x.size(-1)
                    x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
                x = torch.cat([skip, x], dim=2)
            x = dec(x)
            
        # Final conv on last time step's membrane potential
        # x: [T, B, feat, H, W]
        # Read membrane voltages from last neuron if needed
        x = self.final_conv(x)  # spiking conv yields spikes; final conv non-spiking
        _ = self.output_integrator(x)  # integrate spikes to get membrane potential
        v_seq = self.output_integrator.v_seq
        
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

    def double_conv(self, in_channels, out_channels, use_plif=False):
        """
        Helper to create two spiking convolutional layers with batchnorm and LIF/PLIF neurons,
        instantiating fresh neuron nodes for each layer to maintain independent states.
        """
        return nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            self._make_neuron(use_plif=use_plif),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            self._make_neuron(use_plif=use_plif)
        )
