import os
import cv2
import yaml
import shutil
import torchvision
import splitfolders

from tqdm import tqdm
from imageio import imsave
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.util import view_as_windows


def config():
  with open("parameters.yaml", 'r') as stream:
      try:
          parsed_yaml=yaml.safe_load(stream)
      except yaml.YAMLError as exc:
          print("Exception occured when opening .yaml config file!")
  return parsed_yaml


def generate_windows(training_image_path: str, args: dict,
                     img_size: tuple = (128, 128),
                     stride: int = 1):
    """
    Generate sliding windows using scikit-image function `view_as_windows`.

    Parameters
    ----------
    training_image_path : str
        Path to Strebelle training image.
    args : dict
        Parsed arguments as dictionary.
    img_size : tuple, optional
        Size of windows to be saved. The default is (64, 64).
    stride : int, optional
        Step of which the window walks. The default is 1.

    Returns
    -------
    windows : np.ndarray
        Array with batch size containing saved images.
    """
    ti32 = cv2.imread(args["training_image"], cv2.COLOR_BGR2GRAY)

    _, ti = cv2.threshold(ti32, 127, 255, cv2.THRESH_BINARY)

    windows = view_as_windows(ti, (150, 150))
    return windows


def save_generated_images(windowed_images, args: dict) -> None:
    """
    Save each patch of Strebelle image to output directory.

    Parameters
    ----------
    windowed_images : np.ndarray
        Array of images.
    args : dict
        User defined parameters.
    """
    for i, batch_ti in tqdm(enumerate(windowed_images),
                            desc="Sliding window, please wait...",
                            total=windowed_images.shape[0], colour='blue'):
        for j, t in enumerate(batch_ti):
            ti_resized = cv2.resize(t, (128, 128))
            imsave(f"{args['output_dir']}/strebelle_{i}_{j}.png", ti_resized)

def get_strebelle_dataloader(path_to_data='windowed_ti/augmented',
                        batch_size=64):
    """ Augmented Strebelle TI dataloader with (128, 128) sized images. """
    # Get parameters
    param = config()

    # Create necessary folders
    os.makedirs(param["output_dir"], exist_ok=True)

    if not os.path.exists(param["training_image"]):
        raise FileNotFoundError("Could not find Strebelle training image in path!")

    print("Generating sliding windows, please wait...")
    windows = generate_windows(training_image_path=param["training_image"],
                               img_size=128, args=param)

    print("Saving all sliding windows...")
    #save_generated_images(windows, param)

    # Compose transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize([0.5], [0.5])
    ])

    # Splitting data
    #splitfolders.ratio(path_to_data, output="dataset", seed=69069, ratio=(.8, .2))

    # Remove old data
    #shutil.rmtree("windowed_ti/")

    # Get dataset
    train_data = torchvision.datasets.ImageFolder('dataset/train', transform=transform)
    val_data = torchvision.datasets.ImageFolder('dataset/val', transform=transform)

    # Split into train and test dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Create dataloader
    return train_loader, val_loader