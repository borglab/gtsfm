"""Decorator which implements a cache for the two-view estimator class.

Authors: Ayush Baid
"""
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.cache as cache_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.io as io_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.two_view_estimator import TwoViewEstimator, TWO_VIEW_OUTPUT

# Number of first K keypoints from each image to use to create cache key.
NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH = 10

# Number of first K correspondence indices from image pair to use to create cache key.
NUM_CORRESPONDENCES_TO_SAMPLE_FOR_HASH = 10

CACHE_ROOT_PATH = Path(__file__).resolve().parent.parent / "cache"

logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


class TwoViewEstimatorCacher(TwoViewEstimator):
    """Caches two-view relative pose estimation results for an image pair."""

    def __init__(self, two_view_estimator_obj: TwoViewEstimator) -> None:
        self._two_view_estimator = two_view_estimator_obj
        self._verifier_key = self._two_view_estimator._verifier.__repr__()

    def __repr__(self) -> str:
        return self._two_view_estimator.__repr__()

    def __get_cache_path(self, cache_key: str) -> Path:
        """Gets the file path to the cache bz2 file from the cache key."""
        return CACHE_ROOT_PATH / "two_view_estimator" / "{}.pbz2".format(cache_key)

    def __generate_cache_key(
        self, keypoints_i1: Keypoints, keypoints_i2: Keypoints, putative_corr_idxs: np.ndarray
    ) -> str:
        """Generates a cache key according to keypoint coordinates and putative correspondence indices."""
        if putative_corr_idxs.size == 0:  # catch no correspondences
            return cache_utils.generate_hash_for_numpy_array(np.array([]))

        # Subsample correspondence indices.
        sampled_idxs = putative_corr_idxs[:NUM_CORRESPONDENCES_TO_SAMPLE_FOR_HASH]

        # Get the coordinates of the sampled idxs.
        numpy_arrays_to_hash: List[np.ndarray] = []
        numpy_arrays_to_hash.append(keypoints_i1.coordinates[sampled_idxs[:, 0]].flatten())
        numpy_arrays_to_hash.append(keypoints_i2.coordinates[sampled_idxs[:, 1]].flatten())

        # Hash the concatenation of all the numpy arrays.
        input_key = cache_utils.generate_hash_for_numpy_array(np.concatenate(numpy_arrays_to_hash))
        return f"{self._verifier_key}_{input_key}"

    def __load_result_from_cache(
        self, keypoints_i1: Keypoints, keypoints_i2: Keypoints, putative_corr_idxs: np.ndarray
    ) -> Optional[TWO_VIEW_OUTPUT]:
        """Loads cached result, if it exists."""
        cache_key = self.__generate_cache_key(keypoints_i1, keypoints_i2, putative_corr_idxs)
        cache_path = self.__get_cache_path(cache_key=cache_key)
        # If bz2 file does not exist, `None` will be returned.
        cached_data = io_utils.read_from_bz2_file(cache_path)
        return cached_data

    def __save_result_to_cache(
        self, keypoints_i1: Keypoints, keypoints_i2: Keypoints, putative_corr_idxs: np.ndarray, result: TWO_VIEW_OUTPUT
    ) -> None:
        """Saves the result (`TWO_VIEW_OUTPUT` 6-tuple) to the cache."""
        cache_key = self.__generate_cache_key(keypoints_i1, keypoints_i2, putative_corr_idxs)
        cache_path = self.__get_cache_path(cache_key=cache_key)
        io_utils.write_to_bz2_file(result, cache_path)

    def run_2view(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        putative_corr_idxs: np.ndarray,
        camera_intrinsics_i1: Optional[gtsfm_types.CALIBRATION_TYPE],
        camera_intrinsics_i2: Optional[gtsfm_types.CALIBRATION_TYPE],
        i2Ti1_prior: Optional[PosePrior],
        gt_camera_i1: Optional[gtsfm_types.CAMERA_TYPE],
        gt_camera_i2: Optional[gtsfm_types.CAMERA_TYPE],
        gt_scene_mesh: Optional[Any] = None,
    ) -> TWO_VIEW_OUTPUT:
        """Loads 2-view estimation result if it exists in cache, otherwise re-runs two view estimator from scratch."""
        result = self.__load_result_from_cache(keypoints_i1, keypoints_i2, putative_corr_idxs)

        if result is not None:
            return result

        result = self._two_view_estimator.run_2view(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            putative_corr_idxs=putative_corr_idxs,
            camera_intrinsics_i1=camera_intrinsics_i1,
            camera_intrinsics_i2=camera_intrinsics_i2,
            i2Ti1_prior=i2Ti1_prior,
            gt_camera_i1=gt_camera_i1,
            gt_camera_i2=gt_camera_i2,
            gt_scene_mesh=gt_scene_mesh,
        )

        self.__save_result_to_cache(keypoints_i1, keypoints_i2, putative_corr_idxs, result)
        return result
