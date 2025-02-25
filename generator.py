import torch
import torch.nn as nn


class GeneratorConvBlock(nn.Module):
    """
    Defines a convolutional block for the generator.
    Supports both downsampling (Conv2d) and upsampling (ConvTranspose2d).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        down=True,
        use_act=True,
        use_dropout=False,
        **kwargs,
    ):
        """
        Initializes a convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            down (bool): If True, applies downsampling (Conv2d); otherwise, upsampling (ConvTranspose2d).
            use_act (bool): If True, applies ReLU activation; otherwise, uses Identity (no activation).
            use_dropout (bool): If True, applies dropout regularization.
            **kwargs: Additional arguments for Conv2d or ConvTranspose2d.
        """

        super().__init__()
        # Choose between Conv2d (downsampling) and ConvTranspose2d (upsampling)
        if down:
            conv = nn.Conv2d(
                in_channels, out_channels, padding_mode="reflect", **kwargs
            )
        else:
            conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)

        layers = [
            conv,
            nn.InstanceNorm2d(out_channels),
        ]
        if use_act:
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Identity())

        if use_dropout:
            layers.append(nn.Dropout(0.5))  # Dropout for regularization

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class GeneratorResidualBlock(nn.Module):
    """
    Defines a residual block for the generator.
    """

    def __init__(self, channels, use_dropout=False):
        """
        Initializes a residual block.

        Args:
            channels (int): Number of input/output channels (remains unchanged).
            use_dropout (bool): If True, applies dropout in the convolutional layers.
        """
        super().__init__()
        self.res_block = nn.Sequential(
            GeneratorConvBlock(
                channels,
                channels,
                use_act=True,
                kernel_size=3,
                padding=1,
                use_dropout=use_dropout,
            ),
            GeneratorConvBlock(
                channels,
                channels,
                use_act=False,
                kernel_size=3,
                padding=1,
                use_dropout=use_dropout,
            ),
        )

    def forward(self, x):
        return x + self.res_block(x)


class Generator(nn.Module):
    """
    Defines the Generator model in CycleGAN.
    """

    def __init__(
        self, in_channels, num_features=64, num_residuals=9, use_dropout=False
    ):
        """
        Initializes the Generator network.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            num_features (int): Number of feature maps in the first convolutional layer.
            num_residuals (int): Number of residual blocks in the generator.
            use_dropout (bool): If True, applies dropout in residual blocks.
        """

        super().__init__()

        # Initial
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        # Down
        self.down_blocks = nn.ModuleList(
            [
                GeneratorConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                GeneratorConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        # Residual
        res_blocks = []
        for _ in range(num_residuals):
            res_blocks.append(
                GeneratorResidualBlock(num_features * 4, use_dropout=use_dropout)
            )

        self.res_blocks = nn.Sequential(*res_blocks)

        # Up
        self.up_blocks = nn.ModuleList(
            [
                GeneratorConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                GeneratorConvBlock(
                    num_features * 2,
                    num_features,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        # Output
        self.output_layer = nn.Conv2d(
            num_features,
            in_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial_layer(x)
        for block in self.down_blocks:
            x = block(x)

        x = self.res_blocks(x)
        for block in self.up_blocks:
            x = block(x)

        x = self.output_layer(x)
        return torch.tanh(x)
