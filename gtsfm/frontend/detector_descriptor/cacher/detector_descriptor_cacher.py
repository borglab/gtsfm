"""Decorater for caching detector-descriptor output.

Authors: Ayush Baid
"""
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import gtsfm.utils.cache as cache_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase

logger = logger_utils.get_logger()

CACHE_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "cache"


class DetectorDescriptorCacher(DetectorDescriptorBase):
    """Decorator which caches detector-descriptor output, keyed on the input."""

    def __init__(self, detector_descriptor_obj: DetectorDescriptorBase) -> None:
        super().__init__(max_keypoints=detector_descriptor_obj.max_keypoints)
        self._detector_descriptor = detector_descriptor_obj
        # TODO: make the obj cache key dependent on the code
        self._detector_descriptor_obj_cache_key = type(self._detector_descriptor).__name__
        logger.info(self._detector_descriptor_obj_cache_key)

    def __get_cache_path(self, cache_key: str) -> Path:
        return CACHE_ROOT_PATH / "detector_descriptor" / "{}.pkl".format(cache_key)

    def __get_cache_key(self, image: Image) -> str:
        input_key = cache_utils.generate_hash_for_image(image)
        return "{}_{}".format(self._detector_descriptor_obj_cache_key, input_key)

    def __load_result_from_cache(self, image: Image) -> Optional[Tuple[Keypoints, np.ndarray]]:
        cache_path = self.__get_cache_path(cache_key=self.__get_cache_key(image=image))

        if not cache_path.exists():
            return None

        cached_data = pickle.load(open(cache_path, "rb"))

        return cached_data["keypoints"], cached_data["descriptors"]

    def __save_result_to_cache(self, image: Image, keypoints: Keypoints, descriptors: np.ndarray) -> None:
        cache_path = self.__get_cache_path(cache_key=self.__get_cache_key(image=image))

        cache_path.parent.mkdir(exist_ok=True, parents=True)

        cached_data = {"keypoints": keypoints, "descriptors": descriptors}

        pickle.dump(cached_data, open(cache_path, "wb"))

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        cached_data = self.__load_result_from_cache(image)

        if cached_data is not None:
            return cached_data

        keypoints, descriptors = self._detector_descriptor.detect_and_describe(image)
        self.__save_result_to_cache(image, keypoints, descriptors)

        return keypoints, descriptors
