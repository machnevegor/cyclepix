import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from src.discriminator import Discriminator
from torch import nn


def load_discriminator(path, prefix="D_S.", hidden_channels=64):
    checkpoint = torch.load(path, map_location="cpu")

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    clean_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith(prefix):
            new_k = k[len(prefix) :]
            clean_state_dict[new_k] = v

    model = Discriminator(hidden_channels)
    model.load_state_dict(clean_state_dict)
    return model


def show_heatmap(
    heatmap, image_tensor, alpha=0.5, colormap=cv2.COLORMAP_JET, blur=True
):

    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    heatmap = (heatmap * 255).astype(np.uint8)
    if blur:
        heatmap = cv2.GaussianBlur(heatmap, (11, 11), sigmaX=5, sigmaY=5)

    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


def get_tensor_image(img):
    img = np.array(img).astype(np.float32) / 255.0  # [H, W, 3], float32
    img = (img - 0.5) / 0.5  # Нормализация в [-1, 1]
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0)


def show_originals_and_heatmaps(originals, heatmaps, titles=None):
    assert len(originals) == len(heatmaps) == 7, "Нужно ровно 7 изображений"

    plt.figure(figsize=(20, 6))

    for i in range(7):
        plt.subplot(2, 7, i + 1)
        plt.imshow(originals[i])
        plt.axis("off")
        if titles:
            plt.title(titles[i])

        heatmap = heatmaps[i]
        if heatmap.max() <= 1.0:
            heatmap = (heatmap * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor((originals[i]).astype(np.uint8), cv2.COLOR_RGB2BGR), 0.5, heatmap_color, 0.5, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 7, i + 8)
        plt.imshow(overlay)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def tensor_to_numpy_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor * 0.5 + 0.5  # [0,1]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)
