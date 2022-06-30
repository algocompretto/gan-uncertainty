import os
import cv2
import yaml
import time
import torch
import shutil
import numpy as np
import torchvision
import tensorboardX
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from tqdm.auto import tqdm
from imageio import imsave
from torch.autograd import grad
from torch.autograd import Variable
from models.wgan_gp import CriticModel, GeneratorModel


class WGAN_GP:
    """WGAN class to encompass everything."""

    def __init__(self):
        super(WGAN_GP, self).__init__()
        self.args = self.config()
        self.cuda = args["cuda"]
        self.Critic = CriticModel(args["num_channels"])
        self.Generator = GeneratorModel(args["latent_dim"])

    @staticmethod
    def gradient_penalty(x, y, f):
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        alpha = torch.rand(shape).cuda()
        z = x + alpha * (y - x)
        z = Variable(z, requires_grad=True)
        z = z.cuda()
        o = f(z)
        g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(),
                 create_graph=True)[0].view(z.size(0), -1)
        gp = ((g.norm(p=2, dim=1))**2).mean()
        return gp

    @staticmethod
    def save_checkpoint(state, save_path, is_best=False, max_keep=None):
        # Save checkpoint
        torch.save(state, save_path)

        # deal with max_keep
        save_dir = os.path.dirname(save_path)
        list_path = os.path.join(save_dir, 'latest_checkpoint')

        save_path = os.path.basename(save_path)
        if os.path.exists(list_path):
            with open(list_path) as f:
                ckpt_list = f.readlines()
                ckpt_list = [save_path + '\n'] + ckpt_list
        else:
            ckpt_list = [save_path + '\n']

        if max_keep is not None:
            for ckpt in ckpt_list[max_keep:]:
                ckpt = os.path.join(save_dir, ckpt[:-1])
                if os.path.exists(ckpt):
                    os.remove(ckpt)
            ckpt_list[max_keep:] = []

        with open(list_path, 'w') as f:
            f.writelines(ckpt_list)

        # Copy best model
        if is_best:
            shutil.copyfile(save_path,
                            os.path.join(save_dir, 'best_model.ckpt'))

    @staticmethod
    def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
        if os.path.isdir(ckpt_dir_or_file):
            if load_best:
                ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
            else:
                with open(os.path.join(ckpt_dir_or_file,
                                       'latest_checkpoint')) as f:
                    ckpt_path = os.path.join(ckpt_dir_or_file,
                                             f.readline()[:-1])
        else:
            ckpt_path = ckpt_dir_or_file
        ckpt = torch.load(ckpt_path, map_location=map_location)
        print('[INFO] Loading checkpoint from %s succeed!' % ckpt_path)
        return ckpt

    @staticmethod
    def config():
        with open("config/param.yaml", 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError:
                print("Exception occured when opening .yaml config file!")
        return parsed_yaml

    def generate_trivial_augment(self) -> None:
        """ Data augmentation using TrivialAugment """
        ti32 = cv2.imread(self.param["training_image"], cv2.COLOR_BGR2GRAY)
        _, ti = cv2.threshold(ti32, 127, 255, cv2.THRESH_BINARY)

        augmenter = transforms.TrivialAugmentWide()
        imgs = [augmenter(Image.fromarray(np.uint8(ti)))
                for _ in range(30_000)]
        for idx, ti in tqdm(enumerate(imgs)):
            imsave(f"{se;f.param['output_dir']}/strebelle_{idx}.png",
                   np.array(ti))

    def watch_for_checkpoints(self):
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
