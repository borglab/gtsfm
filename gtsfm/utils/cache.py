"""Utility function for caching.

Authors: Ayush Baid
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from gtsfm.common.image import Image

_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "cache"


def cache_root() -> Path:
    """Return the cache directory, or ``GTSFM_CACHE_SUBDIR`` when that variable is set."""
    override = os.environ.get("GTSFM_CACHE_SUBDIR")
    if not override:
        return _DEFAULT_CACHE_ROOT
    path = Path(override)
    return path if path.is_absolute() else _DEFAULT_CACHE_ROOT / path


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
