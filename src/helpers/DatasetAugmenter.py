import os

import numpy as np
import cv2
from src.helpers.funcs import to_binary
from tqdm import tqdm
import albumentations as A
import time

from tempfile import mkdtemp
import os.path as path

def timer(func):
    """
    Times the function passed as argument

    Args:
        func (`function object`): Function which you want to time.
    """
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.4),
    A.GaussianBlur(p=0.6),
    A.ShiftScaleRotate(p=0.5),
    A.GaussNoise(p=0.5),
    A.Cutout(num_holes=10, max_h_size=32, max_w_size=32),
    A.Compose([
    A.OpticalDistortion(0.1, 0.1),
    A.GridDistortion(5, 0.3, 1)
    ])
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
                for i in tqdm(range(10000)):
                    augmented_image = transform(image=image)['image']
                    cv2.imwrite(f"{self.output_dir}/augmented_{image_name.replace('.png', '')}_{i+1}.png",
                                augmented_image)
                    
            except AttributeError as e:
                print(f"Image {image_name} error: {e.args}.")
                pass