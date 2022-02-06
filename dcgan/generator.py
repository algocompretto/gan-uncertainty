import torch.nn as nn
import var


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            # We start with an inverse convolution.
            nn.ConvTranspose2d(var.NZ, var.NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(var.NGF * 8),
            nn.LeakyReLU(True),
            # first batch
            nn.ConvTranspose2d(var.NGF * 8, var.NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(var.NGF * 4),
            nn.LeakyReLU(True),
            # second batch
            nn.ConvTranspose2d(var.NGF * 4, var.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(var.NGF * 2),
            nn.LeakyReLU(True),
            # third batch
            nn.ConvTranspose2d(var.NGF * 2, var.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(var.NGF),
            nn.LeakyReLU(True),
            # fourth and last batch
            nn.ConvTranspose2d(var.NGF, var.NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        """
        The forward pass function for the Generator.

        Arguments:
            input: Takes as argument the input tensor type vector that will be fed to the neural network.
        Returns:
            Returns the output which will be the generated images.
        """
        return self.main(input)
