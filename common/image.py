""" Class for holding an image and its associated data

Authors: Ayush Baid
"""

from typing import Tuple

import numpy as np


class Image:
    """
    Holds the image and associated exif data
    """

    def __init__(self, image_array: np.ndarray, exif_data=None):
        self.image_array = image_array
        self.exif_data = exif_data

    @property
    def shape(self) -> Tuple[int, int]:
        """
        The shape of the image, with the horizontal direction (x coordinate) being the first entry

        Returns:
            Tuple[int, int]: shape of the image
        """
        return self.image_array.shape[1::-1]
