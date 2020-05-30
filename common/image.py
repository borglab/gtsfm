""" Class for holding an image and its associated data

Authors: Ayush Baid
"""

import numpy as np


class Image:
    """
    Holds the image and associated exif data
    """

    def __init__(self, image_array: np.ndarray, exif_data=None):
        self.image_array = image_array
        self.exif_data = exif_data
