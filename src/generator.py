from .resnet import ResNet
from .unet import UNet


class Generator:
    """
    Generator wrapper class that selects and creates a generator model
    based on the provided architecture name ("unet" or "resnet").
    """

    def __init__(
        self, name, hidden_channels, num_resblocks, in_channels=3, out_channels=3
    ):
        self.name = name
        self.hidden_channels = hidden_channels
        self.num_resblocks = num_resblocks
        self.in_channels = in_channels
        self.out_channels = out_channels

    def create(self):
        if self.name == "unet":
            return UNet(self.hidden_channels, self.in_channels, self.out_channels)
        elif self.name == "resnet":
            return ResNet(
                self.hidden_channels,
                self.in_channels,
                self.out_channels,
                self.num_resblocks,
            )
        return "Did not find generator"
