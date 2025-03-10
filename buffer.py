from collections import deque
import torch
import random


class ReplayBuffer:
    """
    A buffer that stores a limited number of images and allows for sampling with replacement.
    """

    def __init__(self, max_size=50):
        """
        Initialize the replay buffer.

        Args:
            max_size (int): Maximum number of images to store in the buffer.
        """
        self.max_size = max_size
        self.imgs = deque(maxlen=max_size)

    def push_and_pop(self, data):
        """
        Add new images to the buffer, replacing existing ones with a probability of 50% when full.

        Args:
            data (Tensor): A batch of images to be added to the buffer.

        Returns:
            Tensor: A batch of images, either newly added or randomly sampled from the buffer.
        """
        to_return = []
        for img in data:
            img = torch.unsqueeze(img, 0)
            if len(self.imgs) < self.max_size:
                self.imgs.append(img)
                to_return.append(img)
            else:
                if random.random() < 0.5:
                    idx = random.randint(0, len(self.imgs) - 1)
                    old_img = self.imgs[idx].clone()
                    self.imgs[idx] = img
                    to_return.append(old_img)
                else:
                    to_return.append(img)
        return torch.cat(to_return, 0)
