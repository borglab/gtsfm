"""
Common utlities for image manipulation

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np


def rgb_to_gray_cv(image: np.ndarray) -> np.ndarray:
    """
    RGB to Grayscale converion using opencv

    Args:
        image (np.array): Input RGB/RGBA image

    Raises:
        ValueError: wrong input dimensions

    Returns:
        np.array: grayscale transformed image
    """

    if len(image.shape) == 2:
        pass
    elif image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
    elif image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        raise ValueError('Input image dimensions are wrong')

    return image
