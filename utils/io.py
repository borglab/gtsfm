"""
Functions to provide I/O APIs for all the modules

Authors: Ayush Baid
"""

import numpy as np
from PIL import Image as PILImage

from common.image import Image


def load_image(img_path: str, scale_factor: float = None) -> Image:
    """
    Load the image from disk

    Args:
        img_path (str): the path of image to load
        scale_factor (float, optional): the multiplicative scaling factor for height and width. Defaults to None.

    Returns:
        np.array: loaded image
    """
    if scale_factor is None:
        return Image(np.asarray(PILImage.open(img_path)))

    highres_img = PILImage.open(img_path)

    width, height = highres_img.size

    small_width = int(width*scale_factor)
    small_height = int(height*scale_factor)

    return Image(np.asarray(highres_img.resize((small_width, small_height))))


def save_image(image: Image, img_path: str):
    """
    Saves the image to disk

    Args:
        image (np.array): image
        img_path (str): the path on disk to save the image to
    """
    im = PILImage.fromarray(image.image_array)
    im.save(img_path)
