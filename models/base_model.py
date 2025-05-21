import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional, monitor

class SpikingUNetRNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 input_size=(128, 128), hidden_dim=4096,
                 encoder_channels=(2, 4)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size  # (H, W)
        self.hidden_dim = hidden_dim
        self.encoder_channels = encoder_channels
      

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

        print(f"Encoder output shape: {self.flat_dim} (c2 * H/4 * W/4)")
        print(f"hidden dim  shape: {hidden_dim} (Recurrent bottleneck)")
        
        # --- Recurrent bottleneck ---
        self.flatten = nn.Flatten(start_dim=2)
        self.recurrent = layer.LinearRecurrentContainer(
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
            in_features=self.flat_dim,
            out_features=self.flat_dim,
            bias=True
        )

        # --- Decoder ---
        self.linear_decoder = nn.Linear(self.flat_dim, self.flat_dim, bias=False)
        self.decoder_neuron = neuron.LIFNode(surrogate_function=surrogate.ATan())

        self.decoder_conv = nn.Sequential(
            layer.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1, bias=False),  # H/2 x W/2
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.ConvTranspose2d(c1, out_channels, kernel_size=4, stride=2, padding=1, bias=False),  # H x W
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

        functional.set_step_mode(self, step_mode='m')
        
        self.output_monitor = monitor.OutputMonitor(self, neuron.LIFNode)
        self.v_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=self, instance=neuron.LIFNode)


        with torch.no_grad():
            self.encoder[0].weight += 0.5 # or another scaling factor
            self.encoder[3].weight += 0.5  # or another scaling factor
            
            self.linear_decoder.weight += 0.5
            
            self.decoder_conv[0].weight += 0.5
            self.decoder_conv[2].weight += 0.5


        # Enable voltage recording
        for m in self.modules():
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = True



    def forward(self, x: torch.Tensor):
        # x: [T, B, in_channels, H, W]
        x = self.encoder(x)                            # [T, B, c2, H/4, W/4]
        x_flat = self.flatten(x)                       # [T, B, flat_dim]
        x_latent = self.recurrent(x_flat)              # [T, B, hidden_dim]

        x_decoded = self.linear_decoder(x_latent)      # [T, B, flat_dim]
        x_decoded = self.decoder_neuron(x_decoded)
        x_reshaped = x_decoded.view(-1, x.shape[1], self.encoder_channels[1],
                                    self.h_down, self.w_down)  # [T, B, c2, H/4, W/4]

        x_out = self.decoder_conv(x_reshaped)          # [T, B, out_channels, H, W]
        
        return x_out
