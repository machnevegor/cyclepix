import numpy as np
import torch


class ReplayBuffer(object):
    """
    Implements a replay buffer to store previously generated images.

    Args:
        max_size (int): Maximum number of images to store in the buffer.
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.current_capacity = 0

    def __call__(self, images):
        """
        Adds new images to the buffer and returns a batch composed of new and/or previously stored images.

        With 50% probability, returns a stored image instead of the new one (if buffer is full),
        replacing it with the new one. Otherwise, returns the new image directly.

        Args:
            images (torch.Tensor): Batch of generated images to possibly store.

        Returns:
            torch.Tensor: Mixed batch of images.
        """
        if self.max_size == 0:
            return images
        imgs = []
        for img in images:
            img = img.unsqueeze(dim=0)

            if self.current_capacity < self.max_size:
                self.current_capacity += 1
                self.buffer.append(img)
                imgs.append(img)
            else:
                p = np.random.uniform(low=0.0, high=1.0)

                if p > 0.5:
                    idx = np.random.randint(low=0, high=self.max_size)
                    tmp = self.buffer[idx].clone()
                    self.buffer[idx] = img
                    imgs.append(tmp)
                else:
                    imgs.append(img)
        return torch.cat(imgs, dim=0) if len(imgs) > 0 else images
