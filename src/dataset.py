import glob

import pytorch_lightning as L
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from .utils import ImageTransform


class DatasetBlock(Dataset):
    """
    A custom dataset class that loads image files and applies transformations.
    """

    def __init__(self, filenames, transform, stage):
        self.filenames = filenames
        self.transform = transform
        self.stage = stage

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = read_image(self.filenames[idx]) / 255.0
        return self.transform(img, stage=self.stage)


class ToStyleModule(L.LightningDataModule):
    """
    PyTorch Lightning data module for style transfer training.
    It prepares paired datasets for photo and style domains.
    """

    def __init__(
        self,
        style_dir,
        photo_dir,
        config,
        sample_size,
        batch_size,
    ):
        super().__init__()
        self.config = config
        self.sample_size = sample_size
        self.batch_size = batch_size

        self.style_filenames = glob.glob(style_dir)
        self.photo_filenames = glob.glob(photo_dir)

        self.transform = ImageTransform()

    def setup(self, stage):
        """
        Setup datasets based on current stage: 'fit', 'test', 'predict'
        """

        if stage == "fit":
            self.style_training = DatasetBlock(
                self.style_filenames, self.transform, stage
            )
            self.photo_training = DatasetBlock(
                self.photo_filenames, self.transform, stage
            )
        if stage in ["fit", "test", "predict"]:
            self.photo_validation = DatasetBlock(
                self.photo_filenames, self.transform, None
            )

    def train_dataloader(self):
        config = {
            "shuffle": True,
            "drop_last": True,
            "batch_size": self.batch_size,
            **self.config,
        }

        style_loader = DataLoader(self.style_training, **config)
        photo_loader = DataLoader(self.photo_training, **config)
        loaders = {"style": style_loader, "photo": photo_loader}

        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return DataLoader(
            self.photo_validation,
            batch_size=self.sample_size,
            **self.config,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return DataLoader(
            self.photo_validation,
            batch_size=self.batch_size,
            **self.config,
        )
