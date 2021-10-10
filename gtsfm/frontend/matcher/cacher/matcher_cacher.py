"""Decorator for caching matcher output.

Authors: Ayush Baid
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import gtsfm.utils.cache as cache_utils
import gtsfm.utils.io as io_utils
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
        """Gets the cache path from the cache key."""
        return CACHE_ROOT_PATH / "matcher" / "{}.pbz2".format(cache_key)

    def __generate_cache_key(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> str:
        """Generates the cache key from the input detections, image shapes, and underlying matcher."""
        numpy_arrays_to_hash: List[np.ndarray] = []

        # subsample and concatenate keypoints and descriptors
        numpy_arrays_to_hash.append(keypoints_i1.coordinates[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        if keypoints_i1.responses is not None:
            numpy_arrays_to_hash.append(keypoints_i1.responses[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        if keypoints_i1.scales is not None:
            numpy_arrays_to_hash.append(keypoints_i1.scales[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        numpy_arrays_to_hash.append(descriptors_i1[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        numpy_arrays_to_hash.append(keypoints_i2.coordinates[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        if keypoints_i2.responses is not None:
            numpy_arrays_to_hash.append(keypoints_i2.responses[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        if keypoints_i2.scales is not None:
            numpy_arrays_to_hash.append(keypoints_i2.scales[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
        numpy_arrays_to_hash.append(descriptors_i2[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())

        # add the shapes as a numpy array
        numpy_arrays_to_hash.append(np.array([im_shape_i1[0], im_shape_i1[1], im_shape_i2[0], im_shape_i2[1]]))

        # hash the concatenation of all the numpy arrays
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
        """Load cached result, if it exists."""
        cache_path = self.__get_cache_path(
            cache_key=self.__generate_cache_key(
                keypoints_i1=keypoints_i1,
                keypoints_i2=keypoints_i2,
                descriptors_i1=descriptors_i1,
                descriptors_i2=descriptors_i2,
                im_shape_i1=im_shape_i1,
                im_shape_i2=im_shape_i2,
            )
        )

        return io_utils.read_from_compressed_file(cache_path)

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
        """Save the results to the cache."""
        cache_path = self.__get_cache_path(
            cache_key=self.__generate_cache_key(
                keypoints_i1=keypoints_i1,
                keypoints_i2=keypoints_i2,
                descriptors_i1=descriptors_i1,
                descriptors_i2=descriptors_i2,
                im_shape_i1=im_shape_i1,
                im_shape_i2=im_shape_i2,
            )
        )
        io_utils.write_to_compressed_file(match_indices, cache_path)

    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> np.ndarray:
        """Match descriptor vectors.

        If the results are in the cache, they are fetched and returned. Otherwise, the `detect_and_describe` of the
        underlying object's API is called and the results are cached.

        # Some matcher implementations (such as SuperGlue) utilize keypoint coordinates as
        # positional encoding, so our matcher API provides them for optional use.

        Output format:
        1. Each row represents a match.
        2. First column represents keypoint index from image #i1.
        3. Second column represents keypoint index from image #i2.
        4. Matches are sorted in descending order of the confidence (score), if possible.

        Args:
            keypoints_i1: keypoints for image #i1, of length N1.
            keypoints_i2: keypoints for image #i2, of length N2.
            descriptors_i1: descriptors corr. to keypoints_i1.
            descriptors_i2: descriptors corr. to keypoints_i2.
            im_shape_i1: shape of image #i1, as (height,width).
            im_shape_i2: shape of image #i2, as (height,width).


        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """
        cached_data = self.__load_result_from_cache(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            descriptors_i1=descriptors_i1,
            descriptors_i2=descriptors_i2,
            im_shape_i1=im_shape_i1,
            im_shape_i2=im_shape_i2,
        )

        if cached_data is not None:
            return cached_data

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
