"""Utility function for caching.

Authors: Ayush Baid
"""

import hashlib

import numpy as np
import torch

from gtsfm.common.image import Image


def generate_hash_for_image(image: Image) -> str:
    """Hash the image using image name, content, and image shape."""
    return hashlib.sha1(
        "{}_{}_{}".format(image.file_name, image.width, image.height).encode()
    ).hexdigest() + generate_hash_for_numpy_array(image.value_array)


def generate_hash_for_numpy_array(input: np.ndarray) -> str:
    """Hash the numpy array."""
    return hashlib.sha1(input.tobytes()).hexdigest()


def generate_hash_for_image_batch(images: torch.Tensor) -> str:
    """Hash a batch of images represented as a torch.Tensor.

    Args:
        images: torch.Tensor of shape (batch_size, channels, height, width) or similar.

    Returns:
        str: SHA1 hash of the batch tensor.
    """
    return hashlib.sha1(images.cpu().numpy().tobytes()).hexdigest()
