from torchvision import transforms
import torch
import os

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_PATH = 'data/train'
VAL_PATH = 'data/val'
NUM_WORKERS = min(4, os.cpu_count())
BATCH_SIZE = 1

NUM_EPOCHS = 3
LR = 2e-4
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 5

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_G_STYLE = "g_style.pth.tar"
CHECKPOINT_G_REAL = "g_real.pth.tar"
CHECKPOINT_CRITIC_STYLE = "critic_style.pth.tar"
CHECKPOINT_CRITIC_REAL = "critic_real.pth.tar"
