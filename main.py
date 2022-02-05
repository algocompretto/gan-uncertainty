"""
Applying covariance to training images
"""
import os
import random

import cv2
import numpy as np
from PIL import Image
import tifffile as tiff
import albumentations as A

original_images_path_list = os.listdir("dataset/training_images/")
os.makedirs("dataset/generated_images/", exist_ok=True)


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


transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.OneOf([
        A.GaussianBlur(),
        A.GaussNoise(),
    ], p=0.3),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.6),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
        A.PiecewiseAffine(p=0.3),
    ], p=0.5),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        A.Emboss(),
        A.RandomBrightnessContrast(),
    ], p=0.4),
    A.HueSaturationValue(p=0.4),
])

ignore_list = ["2D_channels.png", "bangladesh.png", "meandres.png", "strebelle.png", "strebelle_circ.png"]

for image_name in original_images_path_list:
    if image_name not in ignore_list:
        try:
            image = cv2.imread(f"dataset/training_images/{image_name}", 0)

            noise_threshold = random.choice([0.1, 0.3, 0.5,0.7])

            noise_image = sp_noise(image, noise_threshold)

            print(f"Reading and augmenting image: {image_name}")
            cv2.imwrite(f"dataset/generated_images/noise_image_{image_name}.png", noise_image)
            # Applying augmentation with Albumentations
            for i in range(10):
                augmented_image = transform(image=image)['image']
                cv2.imwrite(f"dataset/generated_images/augmented_{image_name}_{i}.png", augmented_image)

        except AttributeError as e:
            print(f"Image {image_name} error, {e.args}.")
