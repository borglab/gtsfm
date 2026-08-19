"""Utility function for caching.

Authors: Ayush Baid
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from gtsfm.common.image import Image


def get_cache_root() -> Path:
    """Root directory for all GTSFM cachers.

    Defaults to <repo>/cache; override with the GTSFM_CACHE_ROOT env var — e.g. to point parallel
    experiment nodes at one shared scratch cache, or to isolate an experiment's cache entirely.
    Read at import time in each process (dask workers inherit the launcher's environment).
    """
    override = os.environ.get("GTSFM_CACHE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent.parent / "cache"


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
