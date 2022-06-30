"""Sliding through Strebelle TI and saving as image."""
import os
import cv2
import argparse
from tqdm import tqdm
from imageio import imsave
from skimage.util import view_as_windows


def config():
    """
    Get parameters for dataset creation.

    Returns
    -------
    dict
        Argument parser.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_image", type=str,
                        default="data/TI/strebelle.png",
                        help="Path to training image")
    parser.add_argument("--output_dir", type=str,
                        default="data/temp/windowed_ti/real/",
                        help="Path to output the training images slices")
    parser.add_argument("--img_size", type=tuple, default=(64, 64),
                        help="Training image window size")
    return parser.parse_args()


def generate_windows(training_image_path: str, args: dict,
                     img_size: tuple = (64, 64),
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

    windows = view_as_windows(ti, args["img_size"])
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
                            desc="Saving each image from batch dimension.",
                            total=windowed_images.shape[0],
                            ncols=100):
        for j, t in enumerate(batch_ti):
            imsave(f"{args['output_dir']}/strebelle_{i}_{j}.png", t)


if __name__ == "__main__":
    # Get parameters
    param = config().__dict__

    # Create necessary folders
    os.makedirs(param["output_dir"], exist_ok=True)

    if not os.path.exists(param["training_image"]):
        raise FileNotFoundError("Could not find Strebelle training image" +
                                "in path!")

    print("Generating sliding windows, please wait...")
    windows = generate_windows(training_image_path=param["training_image"],
                               img_size=param["img_size"], args=param)

    print("Saving all sliding windows...")
    save_generated_images(windows, param)
