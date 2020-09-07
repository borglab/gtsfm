"""Functions to provide I/O APIs for all the modules.

Authors: Ayush Baid
"""

import numpy as np
from PIL import Image as PILImage
from PIL.ExifTags import GPSTAGS, TAGS

from common.image import Image


def load_image(img_path: str) -> Image:
    """Load the image from disk.

    Args:
        img_path (str): the path of image to load.

    Returns:
        Image: loaded image.
    """
    original_image = PILImage.open(img_path)

    exif_data = original_image.getexif()
    if exif_data is not None:
        parsed_data = {}
        for tag, value in exif_data.items():
            if tag in TAGS:
                parsed_data[TAGS.get(tag)] = value
            elif tag in GPSTAGS:
                parsed_data[GPSTAGS.get(tag)] = value
            else:
                parsed_data[tag] = value

        exif_data = parsed_data

    return Image(np.asarray(original_image), exif_data)


def save_image(image: Image, img_path: str):
    """Saves the image to disk

    Args:
        image (np.array): image
        img_path (str): the path on disk to save the image to
    """
    im = PILImage.fromarray(image.image_array)
    im.save(img_path)
