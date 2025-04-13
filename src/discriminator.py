from torch import nn

from .blocks import Downsampling


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator used in CycleGAN.
    Evaluates whether image patches are real or fake.
    """

    def __init__(self, hidden_channels, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            Downsampling(in_channels, hidden_channels, norm=False),  # 64x128x128
            Downsampling(hidden_channels, hidden_channels * 2),  # 128x64x64
            Downsampling(hidden_channels * 2, hidden_channels * 4),  # 256x32x32
            Downsampling(
                hidden_channels * 4, hidden_channels * 8, stride=1
            ),  # 512x31x31
            nn.Conv2d(
                hidden_channels * 8, 1, kernel_size=4, padding=1
            ),  # 1x30x30 (num_channels-h-w)
        )  # 1 channel for binary classification task, 30-30 spatial dimensions of the feature map

    def forward(self, x):
        return self.model(x)
