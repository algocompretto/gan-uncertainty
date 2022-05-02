"""
Applying covariance to training images
"""
import argparse
import os

import albumentations as A
import cv2
from tqdm import tqdm

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


class DatasetAugmenter:
    """
    `DatasetAugmenter`\n
    Is a class to augment data in dataset by applying transforms such
    as Horizontal/Vertical Flip, random rotation by 90 degrees,
    Gaussian blur & noise and random cutouts.
    """
    def __init__(self, images_dir: str, output_dir: str):
        self.original_images_path_list = os.listdir(images_dir)
        self.images_dir = images_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        """
        Function to execute the image transformation and saves to output directory.
        """
        for image_name in self.original_images_path_list:
            try:
                image = cv2.imread(f"{self.images_dir}/{image_name}", 0)
                print(f"[INFO] Reading and augmenting image: {image_name}")
                # Applying augmentation
                for i in tqdm(range(500)):
                    augmented_image = transform(image=image)['image']
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