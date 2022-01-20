import torch
import torchvision
import numpy as np
from torchvision.utils import save_image

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

dataset = torchvision.datasets.DatasetFolder(
        root="data",
        loader = npy_loader,
        extensions=(".npy")
        )

print(dataset)
