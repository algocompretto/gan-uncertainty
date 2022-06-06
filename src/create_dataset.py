"""
Applying covariance to training images
"""
import os
import argparse

from helpers.DatasetAugmenter import DatasetAugmenter

parser = argparse.ArgumentParser()
parser.add_argument("--ti_folder", type=str, default="data/TI", help="training image to be augmented folder path")
parser.add_argument("--output_folder", type=str, default="data/temp/augmented/exp", help="output folder path")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.output_folder, exist_ok=True)
os.makedirs("data/temp/generated_binary/", exist_ok=True)

print("[INFO] Loading and augmenting training images...")
augmenter = DatasetAugmenter(images_dir=opt.ti_folder,
                             output_dir=opt.output_folder)
print("[INFO] Successfully loaded images...")
print("[INFO] Applying augmentation...")
augmenter.run()
