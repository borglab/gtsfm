"""
Wrapper about COLMAP's GRIC estimators, using pycolmap's pybind API.
Uses LORANSAC for Essential, Fundamental, and Homography matrix estimation.

LORANSAC paper:
ftp://cmp.felk.cvut.cz/pub/cmp/articles/matas/chum-dagm03.pdf

On Linux and Mac, a python wheel is available:
https://pypi.org/project/pycolmap/#files
No python wheel is available on Windows.

Authors: John Lambert
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pycolmap
from gtsam import Rot3, Unit3  # type: ignore

import gtsfm.frontend.verifier.verifier_base as verifier_base
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.pycolmap_utils as pycolmap_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.frontend.verifier.verifier_base import VerifierBase

logger = logger_utils.get_logger()


MIN_INLIER_RATIO = 0.01
MIN_NUM_TRIALS = 100000
MAX_NUM_TRIALS = 1000000
CONFIDENCE = 0.999999


class ConfigurationType(Enum):
    UNDEFINED = 0
    # Degenerate configuration (e.g., no overlap or not enough inliers).
    DEGENERATE = 1
    # Essential matrix.
    CALIBRATED = 2
    # Fundamental matrix.
    UNCALIBRATED = 3
    # Homography, planar scene with baseline.
    PLANAR = 4
    # Homography, pure rotation without baseline.
    PANORAMIC = 5
    # Homography, planar or panoramic.
    PLANAR_OR_PANORAMIC = 6
    # Watermark, pure 2D translation in image borders.
    WATERMARK = 7
    # Multi-model configuration, i.e. the inlier matches result from multiple
    # individual, non-degenerate configurations.
    MULTIPLE = 8


class GricVerifier(VerifierBase):
    def __init__(
        self,
        use_intrinsics_in_verification: bool,
        estimation_threshold_px: float,
    ) -> None:
        """Initializes the verifier. GRIC automatically checks E vs. F vs. H inliers.

        (See https://github.com/mihaidusmanu/pycolmap/blob/master/essential_matrix.cc#L98)

        Args:
            use_intrinsics_in_verification: Flag to perform keypoint normalization and compute the essential matrix
                instead of fundamental matrix. This should be preferred when the exact intrinsics are known.
            estimation_threshold_px: maximum distance (in pixels) to consider a match an inlier, under squared
                Sampson distance.
        """
        self._use_intrinsics_in_verification = use_intrinsics_in_verification
        self._estimation_threshold_px = estimation_threshold_px
        self._min_matches = (
            verifier_base.NUM_MATCHES_REQ_E_MATRIX
            if self._use_intrinsics_in_verification
            else verifier_base.NUM_MATCHES_REQ_F_MATRIX
        )

        # for failure, i2Ri1 = None, and i2Ui1 = None, and no verified correspondences, and inlier_ratio_est_model = 0
        self._failure_result = (None, None, np.array([], dtype=np.uint64), 0.0)

    def __estimate_two_view_geometry(
        self,
        uv_i1: np.ndarray,
        uv_i2: np.ndarray,
        camera_intrinsics_i1: CALIBRATION_TYPE,
        camera_intrinsics_i2: CALIBRATION_TYPE,
    ) -> Dict[str, Any]:
        """Use the pycolmap Pybind wrapper to estimate an Essential matrix using LORANSAC.

        Args:
            uv_i1: array of shape (N3,2) representing coordinates of 2d points in image 1.
            uv_i2: array of shape (N3,2) representing corresponding coordinates of 2d points in image 2.
            camera_intrinsics_i1: intrinsics for image i1.
            camera_intrinsics_i2: intrinsics for image i2.

        Returns:
            dictionary containing result status code, estimated relative pose (R,t), and inlier mask.
        """
        result_dict = pycolmap.two_view_geometry_estimation(
            points2D1=uv_i1,
            points2D2=uv_i2,
            camera1=pycolmap_utils.get_pycolmap_camera(camera_intrinsics_i1),
            camera2=pycolmap_utils.get_pycolmap_camera(camera_intrinsics_i2),
            max_error_px=self._estimation_threshold_px,
            min_inlier_ratio=MIN_INLIER_RATIO,
            min_num_trials=MIN_NUM_TRIALS,
            max_num_trials=MAX_NUM_TRIALS,
            confidence=CONFIDENCE,
        )
        return result_dict

    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: CALIBRATION_TYPE,
        camera_intrinsics_i2: CALIBRATION_TYPE,
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
            logger.info("[GRIC] Not enough correspondences for verification.")
            return self._failure_result

        uv_i1 = keypoints_i1.coordinates[match_indices[:, 0]]
        uv_i2 = keypoints_i2.coordinates[match_indices[:, 1]]

        result_dict = self.__estimate_two_view_geometry(uv_i1, uv_i2, camera_intrinsics_i1, camera_intrinsics_i2)

        if not result_dict["success"]:
            logger.info("[GRIC] matrix estimation unsuccessful.")
            return self._failure_result

        logger.info("Two view configuration: %s", ConfigurationType(result_dict["configuration_type"]))

        inlier_ratio_est_model = result_dict["num_inliers"] / match_indices.shape[0]

        inlier_mask = np.array(result_dict["inliers"])
        v_corr_idxs = match_indices[inlier_mask]

        # See https://github.com/colmap/colmap/blob/dev/src/base/pose.h#L72 for quaternion coefficient ordering
        qw, qx, qy, qz = result_dict["qvec"]
        i2Ui1 = result_dict["tvec"]
        i2Ri1 = Rot3(qw, qx, qy, qz)
        i2Ui1 = Unit3(i2Ui1)

        return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio_est_model
