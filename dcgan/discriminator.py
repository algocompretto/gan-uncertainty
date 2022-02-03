import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),  # We normalize all the features along the dimension of the batch.
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),  # We normalize again.
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
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
        output = self.main(input)
        return output.view(-1)
