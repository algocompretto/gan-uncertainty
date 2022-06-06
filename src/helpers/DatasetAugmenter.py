from __future__ import absolute_import
from __future__ import print_function

import os
import time

import albumentations as A
import cv2
import numpy as np
from helpers.funcs import to_binary
from tqdm import tqdm

CONTEXT_LENGTH = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_EPOCH = 72000


class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binarization
        _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        return th

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")


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
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.6),
    A.RandomRotate90(p=0.3),
    A.GaussianBlur(p=0.6),
    A.ShiftScaleRotate(p=0.5),
    A.GaussNoise(p=0.5),
    A.Cutout(num_holes=10, max_h_size=32, max_w_size=32),
    A.Compose([
        A.OpticalDistortion(0.1, 0.1),
        A.GridDistortion(5, 0.3, 1)
    ]),
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
                for i in tqdm(range(100)):
                    augmented_image = transform(image=image)['image']
                    cv2.imwrite(f"{self.output_dir}/augmented_{image_name.replace('.png', '')}_{i + 1}.png",
                                augmented_image)

            except AttributeError as e:
                print(f"Image {image_name} error: {e.args}.")
                pass

    def get_binary(self):
        os.makedirs('data/temp/np', exist_ok=True)
        for f in os.listdir(self.output_dir):
            if f.find(".png") != -1:
                img = Utils.get_preprocessed_img("{}/{}".format(self.output_dir, f), 150)
                file_name = f[:f.find(".png")]

                np.savez_compressed("{}/{}".format('data/temp/np', file_name), features=img)
                retrieve = np.load("{}/{}.npz".format('data/temp/np', file_name))["features"]

                assert np.array_equal(img, retrieve)

        data_all = [np.load('data/temp/np/'+fname) for fname in os.listdir('data/temp/np/')]
        merged_data = {}
        for data in data_all:
            [merged_data.update({k: v}) for k, v in data.items()]
        np.savez('output.npz', **merged_data)
