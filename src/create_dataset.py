"""
Applying covariance to training images
"""
import argparse
import os

import albumentations as A
import cv2
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument("--ti_folder", type=str, default="data/TI", help="training image to be augmented folder path")
parser.add_argument("--output_folder", type=str, default="data/temp/augmented/exp", help="output folder path")
parser.add_argument("--img_size", type=int, default=150, help="size of each image dimension")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.output_folder, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.4),
    A.GaussianBlur(p=0.6),
    A.ShiftScaleRotate(p=0.5),
    A.GaussNoise(p=0.5),
    A.Cutout(num_holes=10, max_h_size=16, max_w_size=16)
])

def sp_noise(image, prob):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        color_space = image.shape[2]
        if color_space == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output


class DatasetAugmenter:
    def __init__(self, images_dir: str, output_dir: str):
        self.original_images_path_list = os.listdir(images_dir)
        self.images_dir = images_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        for image_name in self.original_images_path_list:
            try:
                image = cv2.imread(f"{self.images_dir}/{image_name}", 0)
                print(f"[INFO] Reading and augmenting image: {image_name}")
                # Applying augmentation
                for i in tqdm(range(500)):
                    augmented_image = transform(image=image)['image']
                    #image_resized = resize(augmented_image, (opt.img_size,opt.img_size))
                    cv2.imwrite(f"{self.output_dir}/augmented_{image_name.replace('.png', '')}_{i}.png",
                                augmented_image)

            except AttributeError as e:
                print(f"Image {image_name} error: {e.args}.")
                pass

print("[INFO] Loading and augmenting training images...")
augmenter = DatasetAugmenter(images_dir=opt.ti_folder,
                             output_dir=opt.output_folder)
print("[INFO] Successfully loaded images...")
print("[INFO] Applying augmentation...")
augmenter.run()