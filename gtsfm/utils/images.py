"""Common utilities for image manipulation.

Authors: Ayush Baid
"""
from typing import List, Tuple

import cv2 as cv
import numpy as np
from gtsam import SfmTrack

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
        raise ValueError("Input image dimensions are wrong")

    return Image(output_array, image.exif_data)


def vstack_image_pair(image_i1: Image, image_i2: Image) -> Image:
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


def vstack_image_list(imgs: List[np.ndarray]) -> Image:
    """Concatenate images along a vertical axis and save them.

    Args:
        imgs: list of Images, must all be of same width

    Returns:
        vstack_img: new RGB image, containing vertically stacked images as tiles.
    """
    img_h, img_w, ch = imgs[0].value_array.shape
    assert ch == 3

    # width and number of channels must match
    assert all(img.width == img_w for img in imgs)
    assert all(img.value_array.shape[2] == ch for img in imgs)

    all_heights = [img.height for img in imgs]
    vstack_img = np.zeros((sum(all_heights), img_w, 3), dtype=np.uint8)

    running_h = 0
    for i, img in enumerate(imgs):
        h = img.height
        start = running_h
        end = start + h
        vstack_img[start:end, :, :] = img.value_array
        running_h += h

    return Image(vstack_img)


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

# def get_ax_downsample_factor(img: Image, max_resolution: int) -> Tuple[float, int, int]:
#     """
#     Resizing an image by a specific downsample ratio may not be possible due to lack of a clean
#     divisor. However, we can still determine the exact downsampling ratio.
#     """
#     import pdb; pdb.set_trace()
#     h, w = img.height, img.width
#     shorter_size = min(h, w)
#     if shorter_size == h:
#         new_h = max_resolution
#         scale = new_h / float(h)
#         new_w = int(w * scale)
#     else:
#         new_w = max_resolution
#         scale = new_w / float(w)
#         new_h = int(h * scale)

#     downsampled_img = resize_image(img, new_height=new_h, new_width=new_w)
#     downsample_factor = orig_h / downsampled_img.width
#     target_h = downsampled_img.height
#     target_w = downsampled_img.width
#     return downsample_factor, target_h, target_w


def match_image_widths(
    image_i1: Image, image_i2: Image
) -> Tuple[Image, Image, Tuple[float, float], Tuple[float, float]]:
    """Automatically chooses the target width (larger of the two inputs), and
    scales both images to that width.


    Args:
        image_i1: 1st image to match width.
        image_i2: 2nd image to match width.

    Returns:
        Scaled image_i1.
        Scaled image_i2.
        Scaling factor (W, H) for image_i1.
        Scaling factor (W, H) for image_i2.
    """

    max_width = max(image_i1.width, image_i2.width)

    # scale image_i1
    new_width = int(max_width)
    new_height = int(image_i1.height * new_width / image_i1.width)

    scale_factor_i1 = (new_width / image_i1.width, new_height / image_i1.height)
    scaled_image_i1 = resize_image(image_i1, new_height, new_width)

    # scale image_i2
    new_width = int(max_width)
    new_height = int(image_i2.height * new_width / image_i2.width)

    scale_factor_i2 = (new_width / image_i2.width, new_height / image_i2.height)
    scaled_image_i2 = resize_image(image_i2, new_height, new_width)

    return scaled_image_i1, scaled_image_i2, scale_factor_i1, scale_factor_i2


def get_average_point_color(track: SfmTrack, images: List[Image]) -> Tuple[int, int, int]:
    """
    Args:
        track: 3d point/landmark and its corresponding 2d measurements in various cameras
        images: list of all images for this scene

    Returns:
        r: red color intensity, in range [0,255]
        g: green color intensity, in range [0,255]
        b: blue color intensity, in range [0,255]
    """
    rgb_measurements = []
    for k in range(track.number_measurements()):

        # process each measurement
        i, uv_measured = track.measurement(k)

        u, v = np.round(uv_measured).astype(np.int32)
        # ensure round did not push us out of bounds
        u = np.clip(u, 0, images[i].width - 1)
        v = np.clip(v, 0, images[i].height - 1)
        rgb_measurements += [images[i].value_array[v, u]]

    r, g, b = np.array(rgb_measurements).mean(axis=0).astype(np.uint8)
    return r, g, b
