import math
import os
from typing import Tuple, Any

import cv2
import numpy


def complete_square_image(img, size: int) -> Any:
    """
    Fills image with black stripes to complete the square image shape.

    Args:
        img: image which you want to square
        size: size of crop

    Returns:
        Returns the completed image
    """
    if img.shape[0] < size:
        strip = numpy.zeros((size - img.shape[0], img.shape[1], 3), dtype=numpy.int8)
        img = numpy.vstack((img, strip))

    if img.shape[1] < size:
        strip = numpy.zeros((img.shape[0], size - img.shape[1], 3), dtype=numpy.int8)
        img = numpy.hstack((img, strip))

    return img


def yield_cropped_image(image, crop_size: int = 64, stride: int = 32) -> Tuple[int, int, Any]:
    """
    Generates crops for the input image.
    """
    height = image.shape[0]
    width = image.shape[1]

    y_steps = int(math.ceil(height / stride))
    x_steps = int(math.ceil(width / stride))

    for i in range(y_steps):
        for j in range(x_steps):
            x_min = j * stride
            x_max = min(j * stride + crop_size, width)

            y_min = i * stride
            y_max = min(i * stride + crop_size, height)
            cropped_image = image[y_min:y_max, x_min:x_max]

            if cropped_image.shape[0] != crop_size or cropped_image.shape[1] != crop_size:
                cropped_image = complete_square_image(cropped_image, crop_size)

            yield x_min, y_min, cropped_image


def saves_sliding_windows(image, crop_size: int = 64, stride: int = 16) -> None:
    """
    Slides through image to save each square.

    Args:
        image: image which you want to detect objects from
        crop_size: desired image crop size
        stride: desired stride, the vertical pixel translation
    """
    print("[INFO] Saving in sliding windows...")
    path = r"C:\Users\gustavo.scholze\gan-for-mps\TI_generated"
    try:
        for x_tl, y_tl, crop in yield_cropped_image(image, crop_size=crop_size, stride=stride):
            path_saving = os.path.join(path, f"TI_{x_tl}_{y_tl}.png")
            cv2.imwrite(path_saving, crop)
    except Exception as err:
        print(err.args)


original_TI = cv2.imread(r"C:\Users\gustavo.scholze\gan-for-mps\TI\strebelle.png")

saves_sliding_windows(original_TI)
