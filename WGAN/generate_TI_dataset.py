import math
import os
from typing import Tuple, Any, List, Generator

import cv2
import numpy


# r"C:\Users\gustavo.scholze\gan-for-mps\TI\strebelle.png"
class DatasetCreator:
    def __init__(self, ti_image_path: str, crop_size: int = 64, stride: int = 16) -> None:
        self.original_TI: List[numpy.uint8] = cv2.imread(ti_image_path)
        self.crop_size: int = crop_size
        self.stride: int = stride

    def _complete_square_image(self, size: int = 64) -> Any:
        """
        Fills image with black stripes to complete the square image shape.
        """
        if self.original_TI.shape[0] < size:
            strip = numpy.zeros((size - self.original_TI.shape[0], self.original_TI.shape[1], 3), dtype=numpy.int8)
            img = numpy.vstack((self.original_TI, strip))

        if self.original_TI.shape[1] < size:
            strip = numpy.zeros((self.original_TI.shape[0], size - self.original_TI.shape[1], 3), dtype=numpy.int8)
            img = numpy.hstack((self.original_TI, strip))

        return img

    def _yield_cropped_image(self) -> Generator:
        """
        Generates crops for the input image.
        """
        height = self.original_TI.shape[0]
        width = self.original_TI.shape[1]

        y_steps = int(math.ceil(height / self.stride))
        x_steps = int(math.ceil(width / self.stride))

        for i in range(y_steps):
            for j in range(x_steps):
                x_min = j * self.stride
                x_max = min(j * self.stride + self.crop_size, width)

                y_min = i * self.stride
                y_max = min(i * self.stride + self.crop_size, height)
                cropped_image = self.original_TI[y_min:y_max, x_min:x_max]

                if cropped_image.shape[0] != self.crop_size or cropped_image.shape[1] != self.crop_size:
                    cropped_image = self._complete_square_image(cropped_image, self.crop_size)

                yield x_min, y_min, cropped_image

    def saves_sliding_windows(self) -> None:
        """
        Slides through image to save each square.
        """
        print("[INFO] Saving in sliding windows...")
        path = r"C:\Users\gustavo.scholze\gan-for-mps\TI_generated\data"
        try:
            for x_tl, y_tl, crop in self._yield_cropped_image():
                path_saving = os.path.join(path, f"TI_{x_tl}_{y_tl}.png")
                cv2.imwrite(path_saving, crop)
        except Exception as err:
            print(err.args)
