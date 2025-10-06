"""
Wrapper about COLMAP's LORANSAC Essential matrix estimation, using pycolmap's pybind API.

LORANSAC paper:
ftp://cmp.felk.cvut.cz/pub/cmp/articles/matas/chum-dagm03.pdf

On Linux and Mac, a python wheel is available:
https://pypi.org/project/pycolmap/#files

Authors: John Lambert
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pycolmap
from gtsam import Rot3, Unit3  # type: ignore

import gtsfm.utils.logger as logger_utils
import gtsfm.utils.pycolmap_utils as pycolmap_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.frontend.verifier.verifier_base import VerifierBase

logger = logger_utils.get_logger()


# Default Colmap params.
MIN_INLIER_RATIO = 0.01
MIN_NUM_TRIALS = 1000
MAX_NUM_TRIALS = 10000
CONFIDENCE = 0.9999


class LoRansac(VerifierBase):
    def __init__(
        self,
        use_intrinsics_in_verification: bool,
        estimation_threshold_px: float,
        min_inlier_ratio: float = MIN_INLIER_RATIO,
        min_num_trials: int = MIN_NUM_TRIALS,
        max_num_trials: int = MAX_NUM_TRIALS,
        confidence: float = CONFIDENCE,
    ) -> None:
        """Initializes the verifier.

        (See https://github.com/mihaidusmanu/pycolmap/blob/master/essential_matrix.cc#L98)

        Args:
            use_intrinsics_in_verification: Flag to perform keypoint normalization and compute the essential matrix
                instead of fundamental matrix. This should be preferred when the exact intrinsics are known as opposed
                to approximating them from exif data.
            estimation_threshold_px: maximum distance (in pixels) to consider a match an inlier, under squared
                Sampson distance.
        """
        super().__init__(use_intrinsics_in_verification, estimation_threshold_px)
        self._ransac_options = pycolmap.RANSACOptions(
            {
                "max_error": self._estimation_threshold_px,
                "min_num_trials": min_num_trials,
                "max_num_trials": max_num_trials,
                "min_inlier_ratio": min_inlier_ratio,
                "confidence": confidence,
            }
        )

    def __estimate_essential_matrix(
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
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            dictionary containing result status code, estimated relative pose (R,t), and inlier mask.
        """
        camera_i1: pycolmap.Camera = pycolmap_utils.get_pycolmap_camera(camera_intrinsics_i1)
        camera_i2: pycolmap.Camera = pycolmap_utils.get_pycolmap_camera(camera_intrinsics_i2)

        result_dict = pycolmap.estimate_essential_matrix(
            uv_i1,
            uv_i2,
            camera_i1,
            camera_i2,
            self._ransac_options,
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
            logger.info("[LORANSAC] Not enough correspondences for verification.")
            return self._failure_result

        uv_i1 = keypoints_i1.coordinates[match_indices[:, 0]]
        uv_i2 = keypoints_i2.coordinates[match_indices[:, 1]]

        if self._use_intrinsics_in_verification:
            result_dict = self.__estimate_essential_matrix(uv_i1, uv_i2, camera_intrinsics_i1, camera_intrinsics_i2)
        else:
            result_dict = pycolmap.estimate_fundamental_matrix(uv_i1, uv_i2, self._ransac_options)

        if not result_dict:
            matrix_type = "Essential" if self._use_intrinsics_in_verification else "Fundamental"
            logger.info("[LORANSAC] %s matrix estimation unsuccessful.", matrix_type)
            return self._failure_result

        num_inliers = result_dict["num_inliers"]
        inlier_ratio_est_model = num_inliers / match_indices.shape[0]

        # Backward compatible: support both old 'inliers' and new 'inlier_mask'
        if "inlier_mask" in result_dict:
            # New pycolmap 3.11+
            inlier_mask = np.array(result_dict["inlier_mask"])
        elif "inliers" in result_dict:
            # Old pycolmap <=3.10
            inlier_mask = np.array(result_dict["inliers"])
        else:
            raise KeyError("LoRANSAC result_dict missing both 'inliers' and 'inlier_mask'.")
        v_corr_idxs = match_indices[inlier_mask]
        if self._use_intrinsics_in_verification:
            # case where E-matrix was estimated
            # See https://github.com/colmap/colmap/blob/dev/src/base/pose.h#L72 for quaternion coefficient ordering
            # Note(Ayush): their "cam2_from_cam1" does not mean our i1Ri2.
            qx, qy, qz, qw = result_dict["cam2_from_cam1"].rotation.quat
            i2Ri1 = Rot3(qw, qx, qy, qz)
            i2Ui1 = Unit3(result_dict["cam2_from_cam1"].translation)

        else:
            # case where F-matrix was estimated
            i2Fi1 = result_dict["F"]
            i2Ei1 = verification_utils.fundamental_to_essential_matrix(
                i2Fi1, camera_intrinsics_i1, camera_intrinsics_i2
            )
            (i2Ri1, i2Ui1) = verification_utils.recover_relative_pose_from_essential_matrix(
                i2Ei1=i2Ei1,
                verified_coordinates_i1=uv_i1[inlier_mask],
                verified_coordinates_i2=uv_i2[inlier_mask],
                camera_intrinsics_i1=camera_intrinsics_i1,
                camera_intrinsics_i2=camera_intrinsics_i2,
            )
        return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio_est_model
