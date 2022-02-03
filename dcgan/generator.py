import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            # We start with an inverse convolution.
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            # We normalize all the features along the dimension of the batch.
            nn.BatchNorm2d(512),
            # We apply a ReLU rectification to break the linearity.
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
        )

    def forward(self, input):
        """
        The forward pass function for the Generator.

        Arguments:
            input: Takes as argument the input tensor type vector that will be fed to the neural network.
        Returns:
            Returns the output which will be the generated images.
        """
        output = self.main(input)
        return output
