import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register

@register
class AffordanceBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layer to combine features from all decoder layers
        self.feature_combiner = nn.Linear(input_dim, hidden_dim)

        # Initial convolution to adapt input features
        self.initial_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # Sequence of convolutional and deconvolutional layers
        self.conv_deconv_layers = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ])

        # Final convolution for affordance scoring
        self.final_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        # x shape: [num_layers, batch_size, num_queries, hidden_dim]
        num_layers, batch_size, num_queries, _ = x.shape

        # Combine features from all layers
        x = x.permute(1, 2, 0, 3).contiguous()  # [batch_size, num_queries, num_layers, hidden_dim]
        x = self.feature_combiner(x)  # [batch_size, num_queries, hidden_dim]

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
        x = x.view(batch_size, num_queries, self.output_dim, 256, 256)

        return x