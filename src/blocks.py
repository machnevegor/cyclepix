from torch import nn


class Downsampling(nn.Module):
    """
    A convolutional block for downsampling.

    Consists of:
    - Conv2D
    - Optional InstanceNorm2D
    - Optional activation (LeakyReLU or ReLU)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        stride=2,
        padding=1,
        lrelu=True,
        norm=True,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=not norm,
            )
        )
        if norm:
            self.conv_block.append(nn.InstanceNorm2d(out_channels, affine=True))
        if lrelu is not None:
            self.conv_block.append(nn.LeakyReLU(0.2, True) if lrelu else nn.ReLU(True))

    def forward(self, x):
        return self.conv_block(x)


class Upsampling(nn.Module):
    """
    A transposed convolutional block for upsampling.

    Consists of:
    - ConvTranspose2D
    - InstanceNorm2D
    - Optional Dropout
    - ReLU activation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        stride=2,
        padding=1,
        output_padding=0,
        dropout=False,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
        )

        if dropout:
            self.block.append(nn.Dropout(0.5))
        self.block.append(nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class Residual(nn.Module):
    """
    A residual block using two convolutional sub-blocks.

    Each block consists of:
    - Reflection padding
    - Conv2D + InstanceNorm2D + (optional activation)

    Implements: x + F(x)
    """

    def __init__(self, in_channels, kernel=3, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            Downsampling(
                in_channels,
                in_channels,
                kernel=kernel,
                stride=1,
                padding=0,
                lrelu=False,
            ),
            nn.ReflectionPad2d(padding),
            Downsampling(
                in_channels, in_channels, kernel=kernel, stride=1, padding=0, lrelu=None
            ),
        )

    def forward(self, x):
        return x + self.block(x)
