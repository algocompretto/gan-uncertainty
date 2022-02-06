import torch.nn as nn
import var


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngpu = var.NGPU
        self.main = nn.Sequential(
            # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            nn.Conv2d(var.NUM_CHANNELS, var.NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # second layer
            nn.Conv2d(var.NDF, var.NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(var.NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # third layer
            nn.Conv2d(var.NDF * 2, var.NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(var.NDF * 4),  # We normalize again.
            nn.LeakyReLU(0.2, inplace=True),
            # fourth layer
            nn.Conv2d(var.NDF * 4, var.NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(var.NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # last layer
            nn.Conv2d(var.NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.
        )

    def forward(self, input):
        """
        The forward pass function for the Discriminator.

        Arguments:
            input: Takes as argument the input tensor type vector that will be fed to the neural network.
        Returns:
            Returns the output which will be a value between 0 and 1.
        """
        return self.main(input)
