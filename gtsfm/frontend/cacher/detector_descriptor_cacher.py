"""Cacher for any detector-descriptor, which caches the output on disk in the top level folder `cache`.

This class provides the caching functionality to a GTSFM detector-descriptor. To use this cacher, initialize it with
the detector-descriptor you want to apply the cache on.

Example: To cache output of `SiftDetectorDescriptor`, use 
`DetectorDescriptorCacher(detector_descriptor_obj=SiftDetectorDescriptor())`.

Authors: Ayush Baid
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import gtsfm.utils.cache as cache_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase

logger = logger_utils.get_logger()

CACHE_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "cache"


class DetectorDescriptorCacher(DetectorDescriptorBase):
    """Cacher for detector-descriptor output on disk, keyed on the input."""

    def __init__(self, detector_descriptor_obj: DetectorDescriptorBase) -> None:
        """Initializes the cacher with the actual detector-descriptor object.

        Args:
            detector_descriptor_obj: detector-descriptor to use in case of cache miss.
        """
        super().__init__(max_keypoints=detector_descriptor_obj.max_keypoints)
        self._detector_descriptor = detector_descriptor_obj
        # TODO(ayushbaid): make the obj cache key dependent on the code
        self._detector_descriptor_obj_cache_key = type(self._detector_descriptor).__name__

    def __get_cache_path(self, cache_key: str) -> Path:
        """Gets the file path to the cache bz2 file from the cache key."""
        return CACHE_ROOT_PATH / "detector_descriptor" / "{}.pbz2".format(cache_key)

    def __generate_cache_key(self, image: Image) -> str:
        """Generates the cache key from the input image and underlying detector descriptor."""
        input_key = cache_utils.generate_hash_for_image(image)
        return "{}_{}".format(self._detector_descriptor_obj_cache_key, input_key)

    def __load_result_from_cache(self, image: Image) -> Optional[Tuple[Keypoints, np.ndarray]]:
        """Load cached result, if they exist."""
        cache_path = self.__get_cache_path(cache_key=self.__generate_cache_key(image=image))
        cached_data = io_utils.read_from_bz2_file(cache_path)
        if cached_data is None:
            return None

        # Temporary solution until cache is updated on CI.
        keypoints = cached_data["keypoints"]
        descriptors = cached_data["descriptors"]
        keypoints.descriptors = descriptors

        return keypoints, descriptors

    def __save_result_to_cache(self, image: Image, keypoints: Keypoints, descriptors: np.ndarray) -> None:
        """Save the results to the cache."""
        cache_path = self.__get_cache_path(cache_key=self.__generate_cache_key(image=image))
        data = {"keypoints": keypoints, "descriptors": descriptors}
        io_utils.write_to_bz2_file(data, cache_path)

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description, with caching.

        If the results are in the cache, they are fetched and returned. Otherwise, the `detect_and_describe` of the
        underlying object's API is called and the results are cached.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for
        details about the output format.

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """
        cached_data = self.__load_result_from_cache(image)

        if cached_data is not None:
            return cached_data

        keypoints, descriptors = self._detector_descriptor.detect_and_describe(image)
        self.__save_result_to_cache(image, keypoints, descriptors)

        return keypoints, descriptors
