"""
WGAN with Gradient Penalty training module.

This module trains the WGAN model according to user-specified parameters. If
wanted, you can specify each parameter accordingly.

Example
-------
Usage example::

    $ python3 wgan.py --n_epochs 10_000 --cuda True --batch_size 64

Notes
-----
    Make sure you have CUDA device before setting ``CUDA`` flag as True.
"""

import os
import time
import torch
import argparse
import torchvision
import tensorboardX
import torchvision.transforms as transforms

from torchsummary import summary
from torch.autograd import Variable
from models.wgan_gp import GeneratorModel, CriticModel
from helpers.train_utils import (load_checkpoint, gradient_penalty,
                                 save_checkpoint)


def config():
    """Parameter configuration function.

    Returns
    -------
    argparse.args
        Arguments to be used.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=2,
                        help="Number of epochs to train")
    parser.add_argument("--cuda", type=bool, default=False,
                        help="Boolean for training with/without NVIDIA driver")
    parser.add_argument("--batch_size", default=128,
                        type=int, help="Batch size")
    parser.add_argument("--num_workers", default=4,
                        type=int, help="Number of workers for dataset loading")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Number of epochs to train the Critic module")
    parser.add_argument("--learning_rate", type=float,
                        default=0.0002, help="Learning rate for optimizers")
    parser.add_argument("--latent_dim", type=int,
                        default=100,
                        help="Latent dimension of features to generate")
    parser.add_argument("--images_path", type=str,
                        default="data/temp/windowed_ti/",
                        help="Path to folder with training images")
    parser.add_argument("--num_channels", type=int, default=1,
                        help="Channels in tensor image")
    parser.add_argument("--checkpoint", type=str,
                        default=".checkpoints/wgan_gp",
                        help="Checkpoints directory to save PyTorch models")
    parser.add_argument("--sample_images", type=str,
                        default="sample_images/wgan_gp",
                        help="Output for sampled images")
    return parser.parse_args()


def watch_for_checkpoints(args, Critic, Generator, critic_opt, gen_opt):
    """
    Scan directories to see if there are checkpoints saved.

    Parameters
    ----------
    args : argparse.args
        Parameters defined for model training.
    Critic : nn.Module
        The Critic model to be loaded if any checkpoint is found.
    Generator : nn.Module
        The Critic model to be loaded if any checkpoint is found
    critic_opt : torch.optim.Optimizer
        PyTorch optimizer for the Critic model.
    gen_opt : torch.optim.Optimizer
        PyTorch optimizer for the Generator model.

    Returns
    -------
    start_epoch : int
        The epoch to resume the training, if no checkpoint is found, then
        starts at epoch zero.
    """
    checkpoint = args["checkpoint"]
    save_dir = args["sample_images"]

    # Check if path exists
    if not isinstance(checkpoint, (list, tuple)):
        paths = [checkpoint]
        for path in paths:
            if not os.path.isdir(path):
                os.makedirs(path)
    if not isinstance(save_dir, (list, tuple)):
        paths = [save_dir]
        for path in paths:
            if not os.path.isdir(path):
                os.makedirs(path)
    try:
        # Loads checkpoint and changes state dictionary
        ckpt = load_checkpoint(checkpoint)
        start_epoch = ckpt['epoch']
        Critic.load_state_dict(ckpt['D'])
        Generator.load_state_dict(ckpt['Generator'])
        critic_opt.load_state_dict(ckpt['d_optimizer'])
        gen_opt.load_state_dict(ckpt['g_optimizer'])
    except FileNotFoundError:
        print('[*] No checkpoint!')
        start_epoch = 0

    return start_epoch


def train(args) -> None:
    """
    Train both Generative and Critic model.

    Parameters
    ----------
    args : argparse.args
        Parameters defined for model training.
    """
    Critic = CriticModel(args["num_channels"])
    Generator = GeneratorModel(args["latent_dim"])
    print("Critic detailed description:")
    summary(Critic, (1, args["latent_dim"], args["batch_size"]))
    print("Generator detailed description")
    print(Generator)

    # Instantiates optimizers
    G_opt = torch.optim.Adam(
        Generator.parameters(), lr=args["learning_rate"], betas=(0.5, 0.999))
    C_opt = torch.optim.Adam(
        Critic.parameters(), lr=args["learning_rate"], betas=(0.5, 0.999))

    start_epoch = watch_for_checkpoints(args, Critic, Generator, C_opt, G_opt)

    # Loading Dataset
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Grayscale(num_output_channels=1),
    ])

    writer = tensorboardX.SummaryWriter(".logs/wgan-gp")

    data = torchvision.datasets.ImageFolder(
        args["images_path"], transform=transf)
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=args["batch_size"],
                                             shuffle=True,
                                             num_workers=args["num_workers"])

    # Starting training loop
    for epoch in range(start_epoch, args["n_epochs"]):
        start_time = time.time()
        critic_loss = []
        gen_loss = []
        Generator.train()

        # Sampling from dataloader
        for i, (images, _) in enumerate(dataloader):
            step = epoch * len(dataloader) + i + 1
            images = Variable(images)
            batch = images.size(0)

            if args["cuda"]:
                images = images.cuda()

            # Creates random noise
            z = Variable(torch.randn(batch, args["latent_dim"]))

            if args["cuda"]:
                z = z.cuda()

            # Sends random noise to Generator and gets Critic output
            generated = Generator(z)
            real_criticized = Critic(images)
            fake_criticized = Critic(generated)

            # Compare distance between critic output
            em_distance = real_criticized.mean() - fake_criticized.mean()
            grad_penalty = gradient_penalty(images.data, generated.data,
                                            Critic)

            # Calculates critic loss
            CriticLoss = -em_distance + grad_penalty * 10

            # Append to list for logging purpose
            critic_loss.append(CriticLoss.item())
            Critic.zero_grad()
            CriticLoss.backward()
            C_opt.step()

            # Logs to tensorboard
            writer.add_scalar('Critic/em_dist', em_distance.data.cpu().numpy(),
                              global_step=step)
            writer.add_scalar('Critic/gradient_penalty', grad_penalty.data
                              .cpu().numpy(),
                              global_step=step)
            writer.add_scalar('Critic/critic_loss', CriticLoss.data
                              .cpu().numpy(),
                              global_step=step)

            if step % args["n_critic"] == 0:
                # Random latent noise
                z = Variable(torch.randn(batch, args["latent_dim"]))

                if args["cuda"]:
                    z = z.cuda()

                # Generate new images from this latent vector
                generated = Generator(z)

                fake_criticized = Critic(generated)
                GeneratorLoss = -fake_criticized.mean()

                # Append to list for logging purpose
                gen_loss.append(GeneratorLoss.item())

                # Backward pass
                Critic.zero_grad()
                Generator.zero_grad()
                GeneratorLoss.backward()
                G_opt.step()

                # Logs loss scalar to tensorboard
                writer.add_scalars('Generator',
                                   {"g_loss": GeneratorLoss.data.cpu()
                                    .numpy()}, global_step=step)

                print(
                    f"Epoch {epoch+1} : {i+1}/{len(dataloader)}:" +
                    f"{round((time.time()-start_time)/60, 2)} mins", end='\r')

    print(f"Epoch {epoch+1} completed")

    # Switch to evaluation mode and sample new images
    Generator.eval()

    # Generate new samples to save
    z_sample = Variable(torch.randn(100, args["latent_dim"]))
    if args["cuda"]:
        z_sample = z_sample.cuda()

    fake_gen_images = (Generator(z_sample).data + 1)/2.0

    torchvision.utils.save_image(
        fake_gen_images,
        args["sample_images"]+'/Epoch '+str(epoch+1)+".jpg", nrow=10)

    x = torchvision.utils.make_grid(fake_gen_images, nrow=5)
    writer.add_image("Generated", x, step)

    # Save checkpoints
    save_checkpoint({'epoch': epoch + 1,
                     'D': Critic.state_dict(),
                     'Generator': Generator.state_dict(),
                     'd_optimizer': C_opt.state_dict(),
                     'g_optimizer': G_opt.state_dict()},
                    '%s/Epoch_(%d).ckpt' % (args["checkpoint"], epoch + 1),
                    max_keep=2)


if __name__ == "__main__":
    args = config().__dict__

    train(args)
