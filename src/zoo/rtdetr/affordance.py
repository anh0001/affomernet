import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register

@register
class AffordanceBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_conv_layers, num_deconv_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_conv_layers = num_conv_layers
        self.num_deconv_layers = num_deconv_layers

        # Initial convolution to adapt input features
        self.initial_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)

        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            for _ in range(num_conv_layers)
        ])

        # Deconvolutional layers
        self.deconv_layers = nn.ModuleList()
        for i in range(num_deconv_layers):
            if i < num_deconv_layers - 1:
                self.deconv_layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=8, stride=4, padding=1))
            else:
                # Last deconv layer has different parameters
                self.deconv_layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))

        # Final convolution for affordance scoring
        self.final_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        # x shape: [batch_size, num_queries, 16, 16, input_dim]
        batch_size, num_queries, H, W, C = x.shape
        x = x.view(batch_size * num_queries, C, H, W)

        # Initial convolution
        x = F.relu(self.initial_conv(x))

        # Apply convolutional and deconvolutional layers
        for i in range(self.num_deconv_layers):
            # Apply conv layers before each deconv
            for conv in self.conv_layers[i * 2 : (i + 1) * 2]:
                x = F.relu(conv(x))
            
            # Apply deconv layer
            x = F.relu(self.deconv_layers[i](x))

        # Apply remaining conv layers, if any
        for conv in self.conv_layers[self.num_deconv_layers * 2:]:
            x = F.relu(conv(x))

        # Final convolution for affordance scoring
        x = self.final_conv(x)

        # Reshape output
        output_size = 16 * (4 ** (self.num_deconv_layers - 1)) * 2  # Calculate final size based on number of deconv layers
        x = x.view(batch_size, num_queries, self.output_dim, output_size, output_size)

        return x