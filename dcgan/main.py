from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder

import var
from discriminator import Discriminator
from generator import Generator
from augmentation import DatasetAugmenter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print("[INFO] Loading and augmenting training images...")
augmenter = DatasetAugmenter(images_dir=r"C:\Users\gustavo.scholze\gan-for-mps\TI",
                             output_dir=r"C:\Users\gustavo.scholze\gan-for-mps\TI_augmented\data")
print("[INFO] Successfully loaded images...")
print("[INFO] Applying augmentation...")
augmenter.run()

# Creating the transformations (scaling, tensor conversion, normalization) to apply to the input images.
transform = transforms.Compose([
    transforms.Resize((var.IMAGE_SIZE, var.IMAGE_SIZE)), transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Loading the TI_generated
dataset = ImageFolder(r"C:\Users\gustavo.scholze\gan-for-mps\TI_augmented", transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=var.BATCH_SIZE, shuffle=True,
                                          num_workers=var.NUM_WORKERS)