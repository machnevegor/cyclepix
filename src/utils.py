import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.utils import make_grid


class ImageTransform(object):
    def __init__(self, dim=256):

        self.resize = T.Resize((dim, dim), antialias=True)
        self.train_transform = T.Compose(
            [
                T.Resize((dim, dim), antialias=True),
                T.RandomCrop((dim, dim)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        )

    def __call__(self, image, stage):
        if stage == "fit":
            img = self.train_transform(image)
        else:
            img = self.resize(image)
        return img * 2 - 1  # normalization


def show_img(img_tensor, nrow, title=""):
    img_tensor = img_tensor.detach().cpu() * 0.5 + 0.5
    img_grid = make_grid(img_tensor, nrow=nrow).permute(1, 2, 0)
    plt.figure(figsize=(18, 8))
    plt.imshow(img_grid)
    plt.axis("off")
    plt.title(title)
    plt.show()
