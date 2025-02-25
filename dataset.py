import os

from PIL import Image
from torch.utils.data import Dataset


class ToStyleDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images for style transfer (CycleGAN).

    This dataset handles:
        - Loading images from two directories: one for real images and one for style images.
        - Ensuring that both datasets can be iterated indefinitely using modulo indexing.
        - Applying optional transformations (e.g., resizing, normalization).

    Attributes:
        root_real (str): Path to the directory containing real images.
        root_style (str): Path to the directory containing style images.
        transform (callable, optional): A function/transform to apply to both real and style images.
        real_imgs (list): List of filenames for real images.
        style_imgs (list): List of filenames for style images.
        length (int): The maximum dataset length, ensuring that the dataset does not run out of images.
    """

    def __init__(self, root_real, root_style, transform=None):
        """
        Initializes the dataset by loading image file names from directories.

        Args:
            root_real (str): Path to the directory containing real images.
            root_style (str): Path to the directory containing style images.
            transform (callable, optional): Optional transform to be applied on both images.
        """
        self.root_real = root_real
        self.real_imgs = os.listdir(root_real)

        self.root_style = root_style
        self.style_imgs = os.listdir(root_style)

        self.transform = transform

        self.length = max(len(self.real_imgs), len(self.style_imgs))
        self.real_length = len(self.real_imgs)
        self.style_length = len(self.style_imgs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Use modulo to avoid index errors if one dataset is smaller
        real_img = self.real_imgs[idx % self.real_length]
        real_path = os.path.join(self.root_real, real_img)
        real_img = Image.open(real_path).convert("RGB")

        style_img = self.style_imgs[idx % self.style_length]
        style_path = os.path.join(self.root_style, style_img)
        style_img = Image.open(style_path).convert("RGB")

        if self.transform:
            real_img = self.transform(real_img)
            style_img = self.transform(style_img)

        return real_img, style_img
