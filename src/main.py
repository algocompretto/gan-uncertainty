from helpers.config import parse_args
import torch
from torchvision.datasets import ImageFolder
from wgan_gp import WGAN_GP
import torchvision.transforms as transforms
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args):
    model = WGAN_GP(args)
    trans = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(3, 3)),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3),
        transforms.Normalize([0.5], [0.5])])

    # Load datasets to train and test loaders
    dataset_loaded = ImageFolder(f"data/temp/augmented", transform=trans)
    train_loader = torch.utils.data.DataLoader(dataset_loaded, batch_size=args.batch_size, shuffle=True)

    # Start model training
    model.train(train_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
