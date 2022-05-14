"""Cacher for two-view-estimator, which caches the output on disk in the top level folder `cache`.

Authors: Ayush Baid
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gtsam import Pose3, Rot3, Unit3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.cache as cache_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.two_view_estimator import TwoViewEstimator

logger = logger_utils.get_logger()

CACHE_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "cache"
NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH = 10

ROT_KEY = "rot3"
DIRECTION_KEY = "unit3"
CORR_IDXES_KEY = "idxes"


class TwoViewCacher(TwoViewEstimator):
    """Cacher for TwoViewEstimator's output on disk, keyed on the input."""

    def __init__(self, two_view_estimator: TwoViewEstimator) -> None:
        self._two_view_estimator = two_view_estimator
        # TODO(ayushbaid): make the obj cache key dependent on the code
        self._two_view_estimator_key = (
            type(self._two_view_estimator._matcher).__name__ + type(self._two_view_estimator._verifier).__name__
        )

    def __get_cache_path(self, cache_key: str) -> Path:
        """Gets the file path to the cache bz2 file from the cache key."""
        return CACHE_ROOT_PATH / "two_view" / "{}.pbz2".format(cache_key)

    def __generate_cache_key(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> str:
        """Generates the cache key from the input detections, image shapes, and underlying matcher+verifier.

        This function uses the first few keypoints and descriptors to generate a key, and hence is not guaranteed to
        be unique. However, since its unlikely that keypoints/descriptors exactly match in the selected few indices,
        and differ in the remaining, its a good choice for a key.
        """
        numpy_arrays_to_hash: List[np.ndarray] = []

        for keypoints_i, descriptors_i in zip([keypoints_i1, keypoints_i2], [descriptors_i1, descriptors_i2]):
            # subsample and concatenate keypoints and descriptors
            numpy_arrays_to_hash.append(keypoints_i.coordinates[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
            if keypoints_i.responses is not None:
                numpy_arrays_to_hash.append(keypoints_i.responses[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
            if keypoints_i.scales is not None:
                numpy_arrays_to_hash.append(keypoints_i.scales[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())
            numpy_arrays_to_hash.append(descriptors_i[:NUM_KEYPOINTS_TO_SAMPLE_FOR_HASH].flatten())

        # add the shapes as a numpy array
        h1, w1 = im_shape_i1
        h2, w2 = im_shape_i2
        numpy_arrays_to_hash.append(np.array([h1, w1, h2, w2]))

        # hash the concatenation of all the numpy arrays
        input_key = cache_utils.generate_hash_for_numpy_array(np.concatenate(numpy_arrays_to_hash))

        return "{}_{}".format(self._two_view_estimator_key, input_key)

    def __load_result_from_cache(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        """Load cached result, if it exists. The cached result will be a dictionary with 3 items."""
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
        return io_utils.read_from_bz2_file(cache_path)

    def __save_result_to_cache(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
        data: Dict[str, Any],
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
        io_utils.write_to_bz2_file(data, cache_path)

    def run(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        camera_intrinsics_i1: Optional[gtsfm_types.CALIBRATION_TYPE],
        camera_intrinsics_i2: Optional[gtsfm_types.CALIBRATION_TYPE],
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
        i2Ti1_prior: Optional[PosePrior],
        gt_wTi1: Optional[Pose3],
        gt_wTi2: Optional[Pose3],
        gt_scene_mesh: Optional[Any] = None,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        # TODO: use prior in cache key
        cached_data = self.__load_result_from_cache(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            descriptors_i1=descriptors_i1,
            descriptors_i2=descriptors_i2,
            im_shape_i1=im_shape_i1,
            im_shape_i2=im_shape_i2,
        )

        if cached_data is not None:
            return cached_data[ROT_KEY], cached_data[DIRECTION_KEY], cached_data[CORR_IDXES_KEY]

        i2Ri1, i2Ui1, v_corr_idxs = self._two_view_estimator.run(
            keypoints_i1,
            keypoints_i2,
            descriptors_i1,
            descriptors_i2,
            camera_intrinsics_i1,
            camera_intrinsics_i2,
            im_shape_i1,
            im_shape_i2,
            i2Ti1_prior,
            gt_wTi1,
            gt_wTi2,
            gt_scene_mesh,
        )

        results = {ROT_KEY: i2Ri1, DIRECTION_KEY: i2Ui1, CORR_IDXES_KEY: v_corr_idxs}

        self.__save_result_to_cache(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            descriptors_i1=descriptors_i1,
            descriptors_i2=descriptors_i2,
            im_shape_i1=im_shape_i1,
            im_shape_i2=im_shape_i2,
            data=results,
        )

        return i2Ri1, i2Ui1, v_corr_idxs
