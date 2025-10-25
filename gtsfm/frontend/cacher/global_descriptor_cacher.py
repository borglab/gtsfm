"""Cacher for any global descriptor, which caches the output on disk in the top level folder `cache`.

This class provides the caching functionality to a GTSFM global descriptor. To use this cacher, initialize it with
the global descriptor you want to apply the cache on.

Example: To cache output of `NetVLADGlobalDescriptor`, use
`GlobalDescriptorCacher(global_descriptor_obj=NetVLADGlobalDescriptor())`.

Authors: John Lambert
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision.transforms import v2 as transforms  # type: ignore

import gtsfm.utils.cache as cache_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from gtsfm.loader.loader_base import BatchTransform, ResizeTransform

logger = logger_utils.get_logger()

CACHE_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "cache"


class GlobalDescriptorCacher(GlobalDescriptorBase):
    """Cacher for global_descriptor output on disk, keyed on the input."""

    def __init__(self, global_descriptor_obj: GlobalDescriptorBase) -> None:
        """Initializes the cacher with the actual global_descriptor object.

        Args:
            global_descriptor_obj: global_descriptor to use in case of cache miss.
        """
        self._global_descriptor = global_descriptor_obj
        # TODO(johnwlambert): make the obj cache key dependent on the code
        self._global_descriptor_obj_cache_key = type(self._global_descriptor).__name__

    def __get_cache_path(self, cache_key: str) -> Path:
        """Gets the file path to the cache bz2 file from the cache key."""
        return CACHE_ROOT_PATH / "global_descriptor" / "{}.pbz2".format(cache_key)

    def __generate_cache_key(self, images: torch.Tensor) -> str:
        """Generates the cache key from the input image and underlying global descriptor."""
        input_key = cache_utils.generate_hash_for_image_batch(images)
        # Concatenate class name and image array hash.
        return "{}_{}".format(self._global_descriptor_obj_cache_key, input_key)

    def __load_result_from_cache(self, images: torch.Tensor) -> Optional[list[np.ndarray]]:
        """Load cached result, if they exist."""
        cache_path = self.__get_cache_path(cache_key=self.__generate_cache_key(images=images))
        cached_data = io_utils.read_from_bz2_file(cache_path)
        if cached_data is None:
            return None
        return cached_data["global_descriptor"]

    def __save_result_to_cache(self, images: torch.Tensor, global_descriptors: list[np.ndarray]) -> None:
        """Save the results to the cache."""
        cache_path = self.__get_cache_path(cache_key=self.__generate_cache_key(images=images))
        data = {"global_descriptors": global_descriptors}
        io_utils.write_to_bz2_file(data, cache_path)

    def get_preprocessing_transforms(self) -> tuple[ResizeTransform, BatchTransform | None]:
        """Get preprocessing transforms from the global descriptor, if available.

        Returns:
            A tuple of (ResizeTransform, BatchTransform) or (None, None) if no global descriptor is set.
        """
        if self._global_descriptor is not None:
            return self._global_descriptor.get_preprocessing_transforms()
        else:
            # No global descriptor; return identity transform converts to Tensor
            # This is purely to satisfy the loader's expected interface, because no descriptor will be computed.
            return transforms.Lambda(lambda x: torch.from_numpy(x)), None

    def describe_batch(self, images: torch.Tensor) -> list[np.ndarray]:
        """Perform feature detection as well as their description, with caching.

        If the results are in the cache, they are fetched and returned. Otherwise, the `describe_batch` of the
        underlying object's API is called and the results are cached.

        Refer to describe_batch() in GlobalDescriptorBase for details about the output format.

        Args:
            image: The input image.

        Returns:
            Global image descriptor, of shape (D,).
        """
        cached_data = self.__load_result_from_cache(images)

        if cached_data is not None:
            return cached_data

        global_descriptors = self._global_descriptor.describe_batch(images)
        self.__save_result_to_cache(images, global_descriptors)

        return global_descriptors
