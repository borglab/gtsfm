"""Common utilities for image manipulation.

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

from common.image import Image


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
        raise ValueError("Input image dimensions are wrong")

    return Image(output_array, image.exif_data)


def vstack_images(image_i1: Image, image_i2: Image) -> Image:
    """Vertically stack two images.

    Args:
        image_i1: 1st image to stack.
        image_i2: 2nd image to stack.

    Returns:
        Image: stacked image
    """
    new_height = image_i1.height + image_i2.height
    new_width = max(image_i1.width, image_i2.width)

    stacked_arr = np.ones(
        (new_height, new_width, 3),
        dtype=image_i1.value_array.dtype,
    )

    if np.issubdtype(stacked_arr.dtype, np.integer):
        stacked_arr[:] = 255

    stacked_arr[: image_i1.height, : image_i1.width, :] = image_i1.value_array
    stacked_arr[image_i1.height :, : image_i2.width, :] = image_i2.value_array

    return Image(stacked_arr)


def resize_image(image: Image, new_height: int, new_width: int) -> Image:
    """Resize the image to given dimensions.

    Args:
        image: image to resize.
        new_height: height of the new image.
        new_width: width of the new image.

    Returns:
        resized image.
    """

    resized_value_array = cv.resize(
        image.value_array,
        (new_width, new_height),
        interpolation=cv.INTER_CUBIC,
    )

    return Image(resized_value_array)
