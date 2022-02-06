from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import time

from torch.autograd import Variable
from torchvision.datasets import ImageFolder

import var

from generator import Generator
from discriminator import Discriminator

# Creating the transformations (scaling, tensor conversion, normalization) to apply to the input images.
transform = transforms.Compose([transforms.Resize((var.IMAGE_SIZE, var.IMAGE_SIZE)), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Loading the dataset
dataset = ImageFolder("../dataset", transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=var.BATCH_SIZE, shuffle=True,
                                          num_workers=var.NUM_WORKERS)


def weights_init(m):
    """
    Initialize weights of a neural network.

    Arguments:
        m: Neural network which needs to be initialized.
    """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Creating the generator and discriminator objects
netG = Generator().to(var.DEVICE)
netD = Discriminator().to(var.DEVICE)

if (var.DEVICE.type == "cuda") and (var.NGPU > 1):
    netG = nn.DataParallel(netG, list(range(var.NGPU)))
    netD = nn.DataParallel(netD, list(range(var.NGPU)))

# Apply the weights_init function
netG.apply(weights_init)
netD.apply(weights_init)

os.makedirs("results", exist_ok=True)

fixed_noise = torch.randn(var.IMAGE_SIZE, var.NZ, 1, 1, device=var.DEVICE)
real_label = 1.
fake_label = 0.

# Training the DCGANs

# We create a criterion object that will measure the error between the prediction and the target.
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=var.LEARNING_RATE,
                        betas=(var.BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.001,
                        betas=(var.BETA1, 0.999))

# History of training
img_list = []
G_losses = []
D_losses = []


for epoch in range(var.NUM_EPOCHS):
    for i, data in enumerate(data_loader, 0):
        # 1st Step: Updating the weights of the neural network of the discriminator

        # We initialize to 0 the gradients of the discriminator with respect to  the weights.
        netD.zero_grad()

        real_cpu = data[0].to(var.DEVICE)
        b_size = real_cpu.size(0)
        label = torch.full((b_size), real_label, dtype=torch.float, device=var.DEVICE)

        # Forward pass real batch through discriminator
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)

        # Calculates gradients for Discriminator by backpropagating
        errD_real.backward()
        D_x = output.mean().item()

        # Training with fake batch
        # Generating batch of latent vectors
        noise = torch.randn(b_size, var.NZ, 1, 1, device=var.DEVICE)
        fake = netG(noise)
        label.fill_(fake_label)

        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculating loss
        errD_fake = criterion(output, label)

        errD_fake.backward()

        D_G_z1 = output.mean().item()
        # Summing the gradients
        errD = errD_real + errD_fake
        optimizerD.step()

        # 2nd Step: Updating generator network

        netG.zero_grad()
        label.fill_(real_label)

        output = netD(fake).view(-1)

        # Calculate Generator's loss based on this output
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output().mean().item()

        # Updating Generator
        optimizerG.step()

        # Logging training information
        if i % 1 == 0:
            print(f"[{epoch} {var.NUM_EPOCHS}] [{i} {len(data_loader)}]\t"
                  f"Loss_D:{errD.item()}\tLoss_G:{errG.item()}\t"
                  f"D(x):{D_x}\tD(G(z)):{D_G_z1}/{D_G_z2}")

        G_losses.append(errG.item())
        D_losses.append(errD.item())

