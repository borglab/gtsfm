"""Common utilities for image manipulation.

Authors: Ayush Baid
"""
import cv2 as cv

from gtsfm.common.image import Image


def rgb_to_gray_cv(image: Image) -> Image:
    """
    RGB to Grayscale conversion using opencv

    Args:
        image: Input RGB/RGBA image.

    Raises:
        ValueError: wrong input dimensions

    Returns:
        grayscale transformed image.
    """

    input_array = image.value_array

    output_array = input_array

    if len(input_array.shape) == 2:
        pass
    elif input_array.shape[2] == 4:
        output_array = cv.cvtColor(input_array, cv.COLOR_RGBA2GRAY)
    elif input_array.shape[2] == 3:
        output_array = cv.cvtColor(input_array, cv.COLOR_RGB2GRAY)
    else:
        raise ValueError('Input image dimensions are wrong')

    return Image(output_array, image.exif_data)
