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
        features=(64, 128, 256),
        fc_bottleneck=True,  # New flag to toggle FC bottleneck
        fc_recurrent=True,
        conv_recurrent=False,
        hidden_dim=512,
        output_timesteps=1,
        initial_scaling: int=None,
        use_plif_encoder=False,
        use_plif_recurrent=False,
        use_plif_decoder=False,
        init_tau_recurrent=2.0,
        init_tau_encoder=5.0,
        init_tau_decoder=5.0,
        visualize=False
    ):
        super().__init__()
        self.features = features
        self.input_size = input_size
        self.fc_bottleneck = fc_bottleneck
        self.fc_recurrent = fc_recurrent
        self.conv_recurrent = conv_recurrent
        self.hidden_dim = hidden_dim
        self.output_timesteps = output_timesteps
        self.initial_scaling = initial_scaling
        self.scaled_input_size = (
            input_size[0] // initial_scaling if initial_scaling is not None else input_size[0],
            input_size[1] // initial_scaling if initial_scaling is not None else input_size[1]
        )
        self.use_plif_encoder = use_plif_encoder
        self.use_plif_recurrent = use_plif_recurrent
        self.use_plif_decoder = use_plif_decoder
        self.init_tau_recurrent = init_tau_recurrent
        self.init_tau_encoder = init_tau_encoder
        self.init_tau_decoder = init_tau_decoder
        self.visualize = visualize
        print(f'self fully connected bottleneck: {self.fc_bottleneck}')
        
        # Output scaling and bias parameters
        self.output_scale = nn.Parameter(torch.tensor(5.0))
        self.output_bias = nn.Parameter(torch.tensor(0.5))
        self.recurrent_scale = nn.Parameter(torch.tensor(0.01))
        
        H, W = input_size
        
        depth = len(features)  # Number of downsampling layers
        downscaling_factor = 2 ** (depth - 1) * self.initial_scaling if initial_scaling is not None else 1
        print(f"Input size: {H}x{W}, total downscaling factor: {downscaling_factor}, initial scaling: {self.initial_scaling}")
        assert (
            H % downscaling_factor == 0 
            and W % downscaling_factor == 0
        ), f"Input size {H}x{W} must be divisible by {downscaling_factor} for {len(features)}x stride-2 downsamples."
        
        if conv_recurrent:
            assert len(features) >= 2, "conv_recurrent requires at least two encoder layers"

        if self.initial_scaling is not None:
            self.downscale = layer.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=self.initial_scaling,
                stride=self.initial_scaling,
                padding=0,
                bias=False
            )
            self.upscale = nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=self.initial_scaling,
                stride=self.initial_scaling,
                padding=0,
                bias=False
            )
        else:
            self.downscale = None
            self.upscale = None
            

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        for i, feat in enumerate(features[:-1]):           
            self.encoders.append(self.double_conv(prev_channels, feat, init_tau_encoder, use_plif_encoder))
            self.pools.append(layer.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feat
        
        # Upsampling for recurrent layer    
        if self.conv_recurrent:
            self._recurrent_feedback = torch.zeros(1)  # placeholder
            self.recurrent_upsample = nn.Sequential(
                nn.Upsample(size=self.scaled_input_size, mode='bilinear', align_corners=False),
                nn.Conv2d(features[-2], in_channels, kernel_size=1)  # project channels
            )

        # Bottleneck like in original UNet structure
        self.bottom_channels = features[-1]
        self.bottom_block = self.double_conv(prev_channels, self.bottom_channels, init_tau_encoder, use_plif_encoder)

        # Extra fully connected recurrent bottleneck
        bottom_H = H // downscaling_factor
        bottom_W = W // downscaling_factor
        flat_dim = self.bottom_channels * bottom_H * bottom_W

        if self.fc_bottleneck:
            self.reduce_fc = layer.Linear(flat_dim, hidden_dim, bias=False, step_mode='m')
            print(f"Using fully connected bottleneck with hidden_dim={hidden_dim} and flat_dim={flat_dim}")
            if self.fc_recurrent:
                self.bottleneck_neuron = layer.LinearRecurrentContainer(
                    self._make_neuron(init_tau_recurrent, use_plif=use_plif_recurrent),
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    bias=True
                )
            else:
                self.bottleneck_neuron = self._make_neuron(init_tau=init_tau_recurrent, use_plif=use_plif_recurrent)

            self.expand_fc = layer.Linear(hidden_dim, flat_dim, bias=False, step_mode='m')


        # Decoder path
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_ch = self.bottom_channels
        for feat, skip_ch in zip(reversed(features[:-1]), reversed(features[:-1])):
            # up from prev_ch to feat
            self.upconvs.append(
                layer.ConvTranspose2d(prev_ch, feat, kernel_size=2, stride=2)
            )
            # decoder expects skip_ch + feat channels
            self.decoders.append(
                self.double_conv(skip_ch + feat, feat, init_tau_decoder, use_plif_decoder)
            )
            prev_ch = feat

        # Final 1x1 conv
        self.final_conv = layer.Conv2d(features[0], out_channels, kernel_size=1)
        self.output_integrator = LeakyIntegrator()

        if self.visualize:
            self.output_monitor = monitor.OutputMonitor(self, 
                                                        (neuron.LIFNode, 
                                                         neuron.ParametricLIFNode, 
                                                         layer.LinearRecurrentContainer))
            
            # for m in self.modules():
            #     if isinstance(m, (neuron.LIFNode, neuron.ParametricLIFNode)):
            #         m.store_v_seq = True
                    
        
        # Use single step mode to support recurrent layers
        functional.set_step_mode(self, step_mode='s')

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        # x: [T, B, C, H, W]
        outputs = []

        # Reset all internal states
        for m in self.modules():
            if hasattr(m, 'reset'):
                m.reset()

        if self.conv_recurrent:
            self._recurrent_feedback = torch.zeros(
                x.shape[1], self.features[-2], *self.scaled_input_size, device=x.device
            )

        for t in range(x.shape[0]):
            x_t = x[t]  # [B, C, H, W]
            out_t = self._forward_single_step(x_t)  # single-step forward
            outputs.append(out_t)

        # Stack outputs and aggregate
        v_seq = torch.stack(outputs)  # [T, B, 1, H, W]
        v_agg = self.aggregate_output(v_seq)  # [B, 1, H, W]
        logits = self.output_scale * (v_agg - self.output_bias)

        
        if return_logits:
            return logits
        else:
            probabilities = torch.sigmoid(logits)
            return probabilities

    # True forward method for single step mode
    def _forward_single_step(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        skips = []
        feedback = self._recurrent_feedback if self.conv_recurrent else None

        if self.downscale is not None:
            x = self.downscale(x)
            
        # Concatenate recurrent feedback if enabled
        if self.conv_recurrent:
            fb_scaled = self.recurrent_scale * self.recurrent_upsample(feedback)
            x = x + fb_scaled


        # Encoder
        for idx, (enc, pool) in enumerate(zip(self.encoders, self.pools)):
            x = enc(x)
            if self.conv_recurrent and idx == len(self.encoders) - 1: # last encoder layer (without bottom layer)
                self._recurrent_feedback = x # store feedback from penultimate conv
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottom_block(x)

        if self.fc_bottleneck:
            B, C, H, W = x.shape
            x_flat = x.view(B, -1)
            reduced = self.reduce_fc(x_flat)
            h_mid = self.bottleneck_neuron(reduced)
            x_exp = self.expand_fc(h_mid)
            x = x_exp.view(B, C, H, W)

        # Decoder
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                print(f"Padding skip connection from {skip.shape} to {x.shape}")
                dy = skip.size(-2) - x.size(-2)
                dx = skip.size(-1) - x.size(-1)
                x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        # Final conv + membrane integration
        x = self.final_conv(x)
        _ = self.output_integrator(x)
        v = self.output_integrator.v  # [B, 1, H, W]
        if self.upscale is not None:
            v = self.upscale(v)
        return v    
        

        
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

    
    
    def _make_neuron(self, init_tau = 5.0, use_plif=False):
        if use_plif:
            return neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan()) 
        else:
            return neuron.LIFNode(surrogate_function=surrogate.ATan())

    def double_conv(self, in_channels, out_channels, init_tau = 5.0, use_plif=False):
        """
        Helper to create two spiking convolutional layers with batchnorm and LIF/PLIF neurons,
        instantiating fresh neuron nodes for each layer to maintain independent states.
        """
        return nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            self._make_neuron(init_tau=init_tau, use_plif=use_plif),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            self._make_neuron(init_tau=init_tau, use_plif=use_plif)
        )
