"""Decorator for caching matcher output.

Authors: Ayush Baid
"""
import pickle
from bz2 import BZ2File
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import gtsfm.utils.cache as cache_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase

logger = logger_utils.get_logger()

CACHE_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "cache"
NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH = 10


class MatcherCacher(MatcherBase):
    """Decorator which caches matcher output, keyed on the input."""

    def __init__(self, matcher_obj: MatcherBase) -> None:
        super().__init__()
        self._matcher = matcher_obj
        # TODO: make the obj cache key dependent on the code
        self._matcher_obj_key = type(self._matcher).__name__

    def __get_cache_path(self, cache_key: str) -> Path:
        return CACHE_ROOT_PATH / "matcher" / "{}.pbz2".format(cache_key)

    def __get_cache_key(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> str:
        # subsample and concatenate keypoints and descriptors
        numpy_arrays_to_hash: List[np.ndarray] = []

        # for i1
        numpy_arrays_to_hash.append(keypoints_i1.coordinates[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        if keypoints_i1.responses is not None:
            numpy_arrays_to_hash.append(keypoints_i1.responses[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        if keypoints_i1.scales is not None:
            numpy_arrays_to_hash.append(keypoints_i1.scales[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        numpy_arrays_to_hash.append(descriptors_i1[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())

        # for i2
        numpy_arrays_to_hash.append(keypoints_i2.coordinates[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        if keypoints_i2.responses is not None:
            numpy_arrays_to_hash.append(keypoints_i2.responses[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        if keypoints_i2.scales is not None:
            numpy_arrays_to_hash.append(keypoints_i2.scales[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        numpy_arrays_to_hash.append(descriptors_i2[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())

        # add the shapes as a numpy array
        numpy_arrays_to_hash.append(np.array([im_shape_i1[0], im_shape_i1[1], im_shape_i2[0], im_shape_i2[1]]))

        input_key = cache_utils.generate_hash_for_numpy_array(np.concatenate(numpy_arrays_to_hash))
        return "{}_{}".format(self._matcher_obj_key, input_key)

    def __load_result_from_cache(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        cache_path = self.__get_cache_path(
            cache_key=self.__get_cache_key(
                keypoints_i1=keypoints_i1,
                keypoints_i2=keypoints_i2,
                descriptors_i1=descriptors_i1,
                descriptors_i2=descriptors_i2,
                im_shape_i1=im_shape_i1,
                im_shape_i2=im_shape_i2,
            )
        )

        if not cache_path.exists():
            return None

        return pickle.load(BZ2File(cache_path, "rb"))

    def __save_result_to_cache(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
        match_indices: np.ndarray,
    ) -> None:
        cache_path = self.__get_cache_path(
            cache_key=self.__get_cache_key(
                keypoints_i1=keypoints_i1,
                keypoints_i2=keypoints_i2,
                descriptors_i1=descriptors_i1,
                descriptors_i2=descriptors_i2,
                im_shape_i1=im_shape_i1,
                im_shape_i2=im_shape_i2,
            )
        )
        cache_path.parent.mkdir(exist_ok=True, parents=True)

        pickle.dump(match_indices, BZ2File(cache_path, "wb"))

    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> np.ndarray:
        cached_data = self.__load_result_from_cache(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            descriptors_i1=descriptors_i1,
            descriptors_i2=descriptors_i2,
            im_shape_i1=im_shape_i1,
            im_shape_i2=im_shape_i2,
        )

        if cached_data is not None:
            logger.debug("Cache hit")
            return cached_data

        logger.debug("Cache miss")

        match_indices = self._matcher.match(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            descriptors_i1=descriptors_i1,
            descriptors_i2=descriptors_i2,
            im_shape_i1=im_shape_i1,
            im_shape_i2=im_shape_i2,
        )

        self.__save_result_to_cache(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            descriptors_i1=descriptors_i1,
            descriptors_i2=descriptors_i2,
            im_shape_i1=im_shape_i1,
            im_shape_i2=im_shape_i2,
            match_indices=match_indices,
        )

        return match_indices
