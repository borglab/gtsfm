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

import gtsfm.utils.cache as cache_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase

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

    def __generate_batch_cache_key(self, batch_tensor: torch.Tensor) -> str:
        """Generates a single cache key for an entire batch tensor."""
        # CRITICAL: Move tensor to CPU and convert to NumPy before hashing.
        tensor_hash = cache_utils.generate_hash_for_numpy_array(batch_tensor.cpu().numpy())
        return f"{self._global_descriptor_obj_cache_key}_batch_{tensor_hash}"

    def __load_batch_from_cache(self, batch_tensor: torch.Tensor) -> Optional[List[np.ndarray]]:
        """Load a cached list of descriptors for a whole batch."""
        batch_cache_key = self.__generate_batch_cache_key(batch_tensor)
        cache_path = self.__get_cache_path(batch_cache_key)
        cached_data = io_utils.read_from_bz2_file(cache_path)
        return cached_data["descriptors_list"] if cached_data else None

    def __save_batch_to_cache(self, batch_tensor: torch.Tensor, descriptors: List[np.ndarray]) -> None:
        """Save a list of descriptors for a whole batch."""
        batch_cache_key = self.__generate_batch_cache_key(batch_tensor)
        cache_path = self.__get_cache_path(batch_cache_key)
        data_to_cache = {"descriptors_list": descriptors}
        io_utils.write_to_bz2_file(data_to_cache, cache_path)

    def describe_batch(self, image_batch_tensor: torch.Tensor) -> List[np.ndarray]:
        """Computes descriptors for a batch of images, with 'all-or-nothing' caching."""
        cached_descriptors = self.__load_batch_from_cache(image_batch_tensor)
        if cached_descriptors is not None:
            logger.info("Cache HIT for entire batch.")
            return cached_descriptors

        logger.info(f"Cache MISS for batch. Re-computing {len(image_batch_tensor)} descriptors.")
        new_descriptors = self._global_descriptor.describe_batch(image_batch_tensor)
        self.__save_batch_to_cache(image_batch_tensor, new_descriptors)
        return new_descriptors
