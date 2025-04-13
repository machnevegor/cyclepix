import torch
from torch import nn

from .blocks import Downsampling, Residual, Upsampling


class ResNet(nn.Module):
    def __init__(self, hidden_channels, in_channels, out_channels, num_resblocks):
        super().__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            Downsampling(
                in_channels, hidden_channels, kernel=7, stride=1, padding=0, lrelu=False
            ),  # 64x256x256
            Downsampling(
                hidden_channels, hidden_channels * 2, kernel=3, lrelu=False
            ),  # 128x128x128
            Downsampling(
                hidden_channels * 2, hidden_channels * 4, kernel=3, lrelu=False
            ),  # 256x64x64
            # residual blocks
            *[Residual(hidden_channels * 4) for _ in range(num_resblocks)],  # 256x64x64
            # upsampling path
            Upsampling(
                hidden_channels * 4, hidden_channels * 2, kernel=3, output_padding=1
            ),  # 128x128x128
            Upsampling(
                hidden_channels * 2, hidden_channels, kernel=3, output_padding=1
            ),  # 64x256x256
            nn.ReflectionPad2d(3),  # to handle border pixels
            nn.Conv2d(
                hidden_channels, out_channels, kernel_size=7, stride=1, padding=0
            ),  # 3x256x256
            nn.Tanh(),  # pixels in the range [-1,1]
        )

    def forward(self, x):
        return self.model(x)
