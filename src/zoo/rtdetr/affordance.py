import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register

@register
class AffordanceBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_conv_layers=4, num_deconv_layers=4, conv_kernel_size=3, deconv_kernel_size=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_conv_layers = num_conv_layers
        self.num_deconv_layers = num_deconv_layers
        self.conv_kernel_size = conv_kernel_size
        self.deconv_kernel_size = deconv_kernel_size

        # Feature combiner layer
        # Purpose: Combine features from all decoder layers into a single representation
        # Input: `input_dim` is the number of decoder layers (e.g., 6)
        # Output: `hidden_dim` is the dimension of each feature vector (e.g., 256)
        self.feature_combiner = nn.Linear(input_dim, hidden_dim)
        # How it works:
        # - For each query and spatial location, it transforms a vector of length `input_dim`
        #   (one value from each decoder layer) into a single value.
        # - Essentially performs a learned weighted sum of features from all decoder layers.
        # Advantage:
        # - Allows the network to learn the best way to combine information from different decoder layers.
        # - Can capture both low-level and high-level features from different decoder layers.
        # Alternative approaches could include max pooling or average pooling across layers,
        # but a linear layer provides a learnable and potentially more expressive combination.

        # Initial convolution to adapt input features
        self.initial_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)

        # Sequence of convolutional and deconvolutional layers
        self.conv_deconv_layers = nn.ModuleList()
        for _ in range(num_deconv_layers):
            self.conv_deconv_layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size//2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=deconv_kernel_size, stride=2, padding=deconv_kernel_size//2, output_padding=1),
                nn.ReLU(inplace=True)
            ])
        
        # Add remaining conv layers if any
        for _ in range(num_conv_layers - num_deconv_layers):
            self.conv_deconv_layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size//2),
                nn.ReLU(inplace=True)
            ])

        # Final convolution for affordance scoring
        self.final_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        # x shape: [num_layers, batch_size, num_queries, hidden_dim]
        num_layers, batch_size, num_queries, _ = x.shape

        # Combine features from all layers
        # 1. Rearrange the tensor for the linear layer operation
        x = x.permute(1, 2, 0, 3).contiguous()  # [batch_size, num_queries, num_layers, hidden_dim]
        # 2. Apply the feature combiner to merge information across layers
        x = self.feature_combiner(x)  # [batch_size, num_queries, hidden_dim]
        # The feature_combiner processes each query independently, combining
        # information from all decoder layers into a single feature vector.

        # Reshape to 2D for convolutions
        x = x.view(batch_size * num_queries, self.hidden_dim, 16, 16)  # Assuming 16x16 spatial size

        # Initial convolution
        x = F.relu(self.initial_conv(x))

        # Apply convolutional and deconvolutional layers
        for layer in self.conv_deconv_layers:
            x = layer(x)

        # Final convolution for affordance scoring
        x = self.final_conv(x)

        # Reshape output
        output_size = 16 * (2 ** self.num_deconv_layers)
        x = x.view(batch_size, num_queries, self.output_dim, output_size, output_size)

        return x