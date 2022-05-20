from cmath import inf
from helpers.funcs import generate_images, to_binary
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=7, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=250, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=10, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
parser.add_argument("--output_folder", type=str, default="data/temp/wgan", help="output folder for all of the generated images")
parser.add_argument("--input_folder", type=str, default="data/temp/augmented", help="input folder for all of the augmented images")
opt = parser.parse_args()

# Create directories
os.makedirs(opt.output_folder, exist_ok=True)
os.makedirs("data/logs/", exist_ok=True)


# Summary writer to log losses and images while training
writer = SummaryWriter(log_dir="data/logs/")

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


class Generator(nn.Module):
    """
    `Generator` extending `nn.Module` from PyTorch. \n
    This class generates a random signal and enhances as both networks evolve.\n
    Args:
        nn (nn.Module): The neural network module from PyTorch.
    """
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            """The building blocks for the Generator neural network.

            Args:
                in_feat (`int`): The input features
                out_feat (`int`): Output features, the classes for shale and mud.
                normalize (`bool`, optional): If `True` appends a `BatchNormalization1d` layer
                    to the `Generator` instance. Defaults to True.

            Returns:
                `list`: Returns layers of accordingly.
            """
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, momentum=0.8, eps=1e-05))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers            

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        """The forward pass function to the neural network

        Args:
            z (`Tensor[]`): The tensor containing the latent space signal.

        Returns:
            `Tensor[]` : Returns the image in Tensor format.
        """
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    """
    `Discriminator` extending `nn.Module` from PyTorch. \n
    This class discriminates a random signal and enhances as both networks evolve.\n
    Args:
        nn (nn.Module): The neural network module from PyTorch.
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        """The forward pass function to the neural network

        Args:
            img (`Tensor[]`): The image tensor containing the latent space signal.

        Returns:
            `Tensor[]` : Returns the score of its originality.
        """
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
transform = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(3, 3)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(),
    transforms.RandomErasing(p=0.3),
    transforms.Normalize([0.5], [0.5])])

dataset_loaded = ImageFolder(f"{opt.input_folder}", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset_loaded, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# --------------
#  Training step
# --------------
running_loss=[]
running_epoch_loss=[]
batches_done = 0
best_loss = inf

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
        writer.add_scalar("dis_loss", loss_D, i)
        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))
            if abs(loss_G) < best_loss:
                # Saves the generator network
                torch.save(generator, f"data/Generator.pth")
            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )
            writer.add_scalar("gen_loss", loss_G, i)
        batches_done += 1


generate_images(f"data/Generator.pth",
                imgs.shape[0], opt.latent_dim,
                opt.output_folder)

# Call flush() method to make sure that all pending events have been written to disk.
writer.flush()

# Closes data streams
writer.close()
