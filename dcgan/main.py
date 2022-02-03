from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import cv2

from generator import G
from discriminator import D

# Setting some hyperparameters
batchSize = 1  # We set the size of the batch.
imageSize = 64  # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
                                                                       0.5)), ])  # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = ImageFolder('/content/data/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True,
                                         num_workers=2)  # We use dataLoader to get the images of the training set batch by batch.


# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Creating the generator
netG = G()  # We create the generator object.
netG.apply(weights_init)  # We initialize all the weights of its neural network.


# Creating the discriminator
netD = D() # We create the discriminator object.
netD.apply(weights_init) # We initialize all the weights of its neural network.
