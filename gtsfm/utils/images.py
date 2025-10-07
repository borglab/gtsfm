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


def vstack_image_list(imgs: List[Image]) -> Image:
    """Concatenate images along a vertical axis and save them.

    Args:
        imgs: list of Images, must all be of same width

    Returns:
        vstack_img: new RGB image, containing vertically stacked images as tiles.
    """
    _, img_w, ch = imgs[0].value_array.shape
    assert ch == 3

    # width and number of channels must match
    assert all(img.width == img_w for img in imgs)
    assert all(img.value_array.shape[2] == ch for img in imgs)

    all_heights = [img.height for img in imgs]
    vstack_img = np.zeros((sum(all_heights), img_w, 3), dtype=np.uint8)

    running_h = 0
    for img in imgs:
        h = img.height
        start = running_h
        end = start + h
        vstack_img[start:end, :, :] = img.value_array
        running_h += h

    return Image(vstack_img)


def resize_image(image: Image, new_height: int, new_width: int) -> Image:
    """Resize the image to given dimensions, preserving filename metadata.

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

    # Resize the mask using nearest-neighbor interpolation.
    if image.mask is not None:
        resized_mask = cv.resize(
            image.mask,
            (new_width, new_height),
            interpolation=cv.INTER_NEAREST,
        )
    else:
        resized_mask = None

    return Image(value_array=resized_value_array, file_name=image.file_name, mask=resized_mask)


def resize_to_max_size(img: Image, long_edge_size: int) -> Image:
    """Resizes image such that longest edge is equal to long_edge_size.

    Args:
        img: The input image to be resized.
        long_edge_size: The desired size for the longest edge of the image.

    Returns:
        The resized image.
    """
    max_size = max(img.height, img.width)
    ratio = float(long_edge_size) / max_size
    new_height = int(img.height * ratio)
    new_width = int(img.width * ratio)
    new_image = resize_image(img, new_height, new_width)
    return new_image


def get_rescaling_factor_per_axis(img_h: int, img_w: int, max_resolution: int) -> Tuple[float, float, int, int]:
    """Resize an image such that the shorter image side is *exactly equal* to max_resolution.

    Note: this may involve downsampling OR upsampling the image.

    Resizing an image by a specific downsample ratio may not be possible due to lack of a clean
    divisor. However, we can still determine the exact downsampling ratio.

    Args:
        img_h: height of image to be resized, in pixels
        img_w: width of image to be resized, in pixels
        max_resolution: integer representing length of image's short side
            e.g. for 1080p (1920 x 1080), max_resolution would be 1080

    Returns:
        scale_u: rescaling factor for u coordinate
        scale_v: rescaling factor for v coordinate. May not be equal to scale_u due to integer-rounding.
        new_h: new height that will preserve aspect ratio as closely as possible, while
            respecting max_resolution constraint
        new_w: new width
    """
    h, w = img_h, img_w
    shorter_size = min(h, w)
    if shorter_size == h:
        new_h = max_resolution
        # compute scaling that will be applied to original image
        scale = new_h / float(h)
        new_w = np.round(w * scale).astype(np.int32)
    else:
        new_w = max_resolution
        scale = new_w / float(w)
        new_h = np.round(h * scale).astype(np.int32)

    scale_u = new_w / w
    scale_v = new_h / h

    return scale_u, scale_v, new_h, new_w


def get_downsampling_factor_per_axis(img_h: int, img_w: int, max_resolution: int) -> Tuple[float, float, int, int]:
    """Resize an image such that the shorter image side is *less than or equal to* the max_resolution.

    This will always downsample the image or leave it intact.

    Note: Different from COLMAP's `Downsize()`, which instead checks if ANY side is
        larger than the max resolution.
        See: https://github.com/colmap/colmap/blob/dev/src/mvs/image.cc#L83

    Args:
        img_h: height of image to be downsampled, in pixels
        img_w: width of image to be downsampled, in pixels
        max_resolution: integer representing maximum length of image's short side
            e.g. for 1080p (1920 x 1080), max_resolution would be 1080

    Returns:
        scale_u: rescaling factor for u coordinate
        scale_v: rescaling factor for v coordinate. May not be equal to scale_u due to integer-rounding.
        new_h: new height that will preserve aspect ratio as closely as possible, while
            respecting max_resolution constraint
        new_w: new width
    """
    if min(img_h, img_w) > max_resolution:
        scale_u, scale_v, new_h, new_w = get_rescaling_factor_per_axis(img_h, img_w, max_resolution)

    else:
        scale_u = 1.0
        scale_v = 1.0
        new_h = img_h
        new_w = img_w

    return scale_u, scale_v, new_h, new_w


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
    """Computes the average point color over all measurements in a track.

    Args:
        track: 3d point/landmark and its corresponding 2d measurements in various cameras.
        images: List of all images for this scene.

    Returns:
        r: Red color intensity, in range [0,255].
        g: Green color intensity, in range [0,255].
        b: Blue color intensity, in range [0,255].
    """
    rgb_measurements = []
    for k in range(track.numberMeasurements()):

        # process each measurement
        i, uv_measured = track.measurement(k)

        u, v = np.round(uv_measured).astype(np.int32)
        # ensure round did not push us out of bounds
        u = np.clip(u, 0, images[i].width - 1)
        v = np.clip(v, 0, images[i].height - 1)
        rgb_measurements += [images[i].value_array[v, u]]

    r, g, b = np.array(rgb_measurements).mean(axis=0).astype(np.uint8)
    return r, g, b
