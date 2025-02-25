import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """
    Defines a convolutional block for the Discriminator.
    """

    def __init__(self, in_channels, out_channels, stride=2):
        """
        Initializes a Discriminator block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for convolution (default: 2).
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class Discriminator(nn.Module):
    """
    Defines the Discriminator in CycleGAN.
    """

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels_ = features[0]

        for feature in features[1:]:
            stride = 1
            if feature == features[-1]:
                stride = 2

            layers.append(DiscriminatorBlock(in_channels_, feature, stride=stride))
            in_channels_ = feature

        layers.append(
            nn.Conv2d(
                in_channels_,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )

        self.d = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.d(x)
        return torch.sigmoid(x)
