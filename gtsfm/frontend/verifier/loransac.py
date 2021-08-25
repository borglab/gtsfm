
"""
Wrapper about COLMAP's LORANSAC Essential matrix estimation, using pycolmap's pybind API.

On Linux, a python wheel is available:
(add URL)

On Mac, no wheel is available, so system-wide installation of COLMAP is required using this module.
   Follow the instructions first here: https://colmap.github.io/install.html
   and then here: https://github.com/mihaidusmanu/pycolmap#getting-started

Authors: John Lambert
"""

from typing import Optional, Tuple

import numpy as np
import pycolmap
from gtsam import Cal3Bundler, Rot3, Unit3
from scipy.spatial.transform import Rotation


import gtsfm.utils.features as feature_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.verifier_base import VerifierBase, NUM_MATCHES_REQ_E_MATRIX, NUM_MATCHES_REQ_F_MATRIX


logger = logger_utils.get_logger()


class LoRansac(VerifierBase):
    def __init__(
        self, use_intrinsics_in_verification: bool, estimation_threshold_px: float, min_allowed_inlier_ratio_est_model: float
    ) -> None:
        """Initializes the verifier.

        Note: LoRANSAC is hard-coded in pycolmap to use the following hyperparameters:
            min_inlier_ratio = 0.01
            min_num_trials = 1000
            max_num_trials = 100000
            confidence = 0.9999

        (See https://github.com/mihaidusmanu/pycolmap/blob/master/essential_matrix.cc#L98)

        Args:
            use_intrinsics_in_verification: Flag to perform keypoint normalization and compute the essential matrix
                instead of fundamental matrix. This should be preferred when the exact intrinsics are known as opposed
                to approximating them from exif data.
            estimation_threshold_px: maximum distance (in pixels) to consider a match an inlier, under squared
                Sampson distance.
            min_allowed_inlier_ratio_est_model: minimum allowed inlier ratio w.r.t. the estimated model to accept
                the verification result and use the image pair, i.e. the lowest allowed ratio of
                #final RANSAC inliers/ #putatives. A lower fraction indicates less agreement among the result.
        """
        self._use_intrinsics_in_verification = use_intrinsics_in_verification
        self._estimation_threshold_px = estimation_threshold_px
        self._min_allowed_inlier_ratio_est_model = min_allowed_inlier_ratio_est_model
        self._min_matches = (
            NUM_MATCHES_REQ_E_MATRIX
            if self._use_intrinsics_in_verification
            else NUM_MATCHES_REQ_F_MATRIX
        )

        # for failure, i2Ri1 = None, and i2Ui1 = None, and no verified correspondences, and inlier_ratio_est_model = 0
        self._failure_result = (None, None, np.array([], dtype=np.uint64), 0.0)

    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, float]:
        """Performs verification of correspondences between two images to recover the relative pose and indices of
        verified correspondences.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3. These are subset of match_indices.
            Inlier ratio of w.r.t. the estimated model, i.e. the #final RANSAC inliers/ #putatives.
        """
        if match_indices.shape[0] < self._min_matches:
            logger.info('[LORANSAC] Not enough correspondences for verification.')
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
            points2D1, points2D2, camera_dict1, camera_dict2, max_error_px=self._estimation_threshold_px
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

        inlier_ratio_est_model = num_inliers / match_indices.shape[0]
        if inlier_ratio_est_model >= self._min_allowed_inlier_ratio_est_model:
            i2Ri1 = Rot3(Rotation.from_quat([qx,qy,qz,qw]).as_matrix())
            i2Ui1 = Unit3(i2Ui1)
            inlier_mask = result_dict["inliers"]
            v_corr_idxs = match_indices[inlier_mask]
        else:
            i2Ri1 = None
            i2Ui1 = None
            v_corr_idxs = np.array([])
        
        return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio_est_model
