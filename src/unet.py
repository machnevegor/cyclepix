import torch
from torch import nn

from .blocks import Downsampling, Upsampling


class UNet(nn.Module):
    def __init__(self, hidden_channels, in_channels, out_channels):
        super().__init__()

        self.downsampling_block = nn.Sequential(
            Downsampling(
                in_channels, hidden_channels, norm=False
            ),  # 64x128x128 out_channels-height-width
            Downsampling(hidden_channels, hidden_channels * 2),  # 128x64x64
            Downsampling(hidden_channels * 2, hidden_channels * 4),  # 256x32x32
            Downsampling(hidden_channels * 4, hidden_channels * 8),  # 512x16x16
            Downsampling(hidden_channels * 8, hidden_channels * 8),  # 512x8x8
            Downsampling(hidden_channels * 8, hidden_channels * 8),  # 512x4x4
            Downsampling(hidden_channels * 8, hidden_channels * 8),  # 512x2x2
            Downsampling(hidden_channels * 8, hidden_channels * 8, norm=False),
            # 512x1x1, instance norm does not work on 1x1
        )

        self.upsampling_block = nn.Sequential(
            Upsampling(
                hidden_channels * 8, hidden_channels * 8, dropout=True
            ),  # (512+512)x2x2
            Upsampling(
                hidden_channels * 16, hidden_channels * 8, dropout=True
            ),  # (512+512)x4x4
            Upsampling(
                hidden_channels * 16, hidden_channels * 8, dropout=True
            ),  # (512+512)x8x8
            Upsampling(hidden_channels * 16, hidden_channels * 8),  # (512+512)x16x16
            Upsampling(hidden_channels * 16, hidden_channels * 4),  # (256+256)x32x32
            Upsampling(hidden_channels * 8, hidden_channels * 2),  # (128+128)x64x64
            Upsampling(hidden_channels * 4, hidden_channels),  # (64+64)x128x128
        )

        self.feature_block = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 2, out_channels, kernel_size=4, stride=2, padding=1
            ),  # 3x256x256
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []

        for down in self.downsampling_block:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for up, skip in zip(self.upsampling_block, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        return self.feature_block(x)
