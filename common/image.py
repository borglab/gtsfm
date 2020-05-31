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

    def get_shape(self) -> Tuple[int]:
        """
        Returns the shape of the image as a list

        Returns:
            List[int]: the shape in the format [horizontal length (W), vertical length (H)]
        """
        return self.image_array.shape[1::-1]
