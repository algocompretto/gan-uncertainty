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



# Defining the generator
class G(nn.Module):  # We introduce a class to define the generator.

    def __init__(self):  # We introduce the __init__() function that will define the architecture of the generator.
        super(G, self).__init__()  # We inherit from the nn.Module tools.
        self.main = nn.Sequential(
            # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),  # We start with an inversed convolution.
            nn.BatchNorm2d(512),  # We normalize all the features along the dimension of the batch.
            nn.ReLU(True),  # We apply a ReLU rectification to break the linearity.
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # We add another inversed convolution.
            nn.BatchNorm2d(256),  # We normalize again.
            nn.ReLU(True),  # We apply another ReLU.
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # We add another inversed convolution.
            nn.BatchNorm2d(128),  # We normalize again.
            nn.ReLU(True),  # We apply another ReLU.
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # We add another inversed convolution.
            nn.BatchNorm2d(64),  # We normalize again.
            nn.ReLU(True),  # We apply another ReLU.
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # We add another inversed convolution.
            nn.Tanh()  # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
        )

    def forward(self,
                input):  # We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output containing the generated images.
        output = self.main(
            input)  # We forward propagate the signal through the whole neural network of the generator defined by self.main.
        return output  # We return the output containing the generated images.

