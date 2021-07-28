
from typing import Optional, Tuple

import cv2
import numpy as np
import pycolmap
from gtsam import Cal3Bundler, Rot3, Unit3
from scipy.spatial.transform import Rotation


import gtsfm.utils.features as feature_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.verifier_base import VerifierBase, NUM_MATCHES_REQ_E_MATRIX, NUM_MATCHES_REQ_F_MATRIX


DEFAULT_RANSAC_SUCCESS_PROB = 0.99999
DEFAULT_RANSAC_MAX_ITERS = 20000
MAX_TOLERATED_POLLUTION_INLIER_RATIO_EST_MODEL = 0.1

logger = logger_utils.get_logger()


class LoRansac(VerifierBase):
    def __init__(
        self, use_intrinsics_in_verification: bool, estimation_threshold_px: float
    ) -> None:
        """Initializes the verifier.

        Args:
        use_intrinsics_in_verification: Flag to perform keypoint normalization and compute the essential matrix
                instead of fundamental matrix. This should be preferred when the exact
                intrinsics are known as opposed to approximating them from exif data.
        estimation_threshold_px: epipolar distance threshold (measured in pixels)
        """
        self._use_intrinsics_in_verification = use_intrinsics_in_verification
        self._px_threshold = estimation_threshold_px
        self._min_matches = (
            NUM_MATCHES_REQ_E_MATRIX
            if self._use_intrinsics_in_verification
            else NUM_MATCHES_REQ_F_MATRIX
        )

        # for failure, i2Ri1 = None, and i2Ui1 = None, and no verified correspondences, and inlier_ratio_est_model = 0
        self._failure_result = (None, None, np.array([], dtype=np.uint64), 0)

    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, float]:
        """ """
        # return if not enough matches
        if match_indices.shape[0] < self._min_matches:
            logger.info('[LORANSAC] Not enough matches for verification.')
            return self._failure_result 

        uv_i1 = keypoints_i1.coordinates
        uv_i2 = keypoints_i2.coordinates

        focal_length = camera_intrinsics_i1.fx()
        cx, cy = camera_intrinsics_i1.px(), camera_intrinsics_i1.py()

        # TODO: use more accurate proxy?
        width = int(cx*2)
        height = int(cy*2)

        camera_dict1 = {
            "model": "SIMPLE_PINHOLE",
            "width": width,
            "height": height,
            "params": [focal_length, cx, cy],
        }

        focal_length = camera_intrinsics_i2.fx()
        cx, cy = camera_intrinsics_i2.px(), camera_intrinsics_i2.py()
        camera_dict2 = {
            "model": "SIMPLE_PINHOLE",
            "width": width,
            "height": height,
            "params": [focal_length, cx, cy],
        }

        points2D1 = uv_i1[match_indices[:, 0]]
        points2D2 = uv_i2[match_indices[:, 1]]

        result_dict = pycolmap.essential_matrix_estimation(
            points2D1, points2D2, camera_dict1, camera_dict2, max_error_px=4.0
        )

        success = result_dict["success"]
        if not success:
            logger.info('[LORANSAC] Essential matrix estimation unsuccessful.')
            return self._failure_result
        E = result_dict["E"]
        # See https://github.com/colmap/colmap/blob/dev/src/base/pose.h#L72
        qw, qx, qy, qz = result_dict["qvec"]
        i2Ui1 = result_dict["tvec"]
        num_inliers = result_dict["num_inliers"]
        inlier_mask = result_dict["inliers"]

        i2Ri1 = Rot3(Rotation.from_quat([qx,qy,qz,qw]).as_matrix())
        i2Ui1 = Unit3(i2Ui1)

        inlier_ratio_est_model = num_inliers / match_indices.shape[0]

        v_corr_idxs = match_indices[inlier_mask]

        return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio_est_model
