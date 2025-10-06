"""Base class for the V (verification) stage of the frontend.

Authors: Ayush Baid, John Lambert
"""

import abc
from typing import Optional, Tuple

import numpy as np
from gtsam import Rot3, Unit3  # type: ignore

from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata

NUM_MATCHES_REQ_E_MATRIX = 5
NUM_MATCHES_REQ_F_MATRIX = 8


class VerifierBase(GTSFMProcess):
    """Base class for all verifiers.

    Verifiers take the coordinates of the matches as inputs and returns the estimated essential matrix as well as
    geometrically verified points.
    """

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="Verifier",
            input_products=("Keypoints", "Putative Correspondences", "Camera Intrinsics"),
            output_products=("Relative Rotation", "Relative Translation", "Verified Correspondences"),
            parent_plate="Two-View Estimator",
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            + f"__use_intrinsics{self._use_intrinsics_in_verification}_{self._estimation_threshold_px}px"
        )

    def __init__(
        self,
        use_intrinsics_in_verification: bool,
        estimation_threshold_px: float,
    ) -> None:
        """Initializes the verifier.

        Args:
            use_intrinsics_in_verification: Flag to perform keypoint normalization and compute the essential matrix
                instead of fundamental matrix. This should be preferred when the exact intrinsics are known as opposed
                to approximating them from exif data.
            estimation_threshold_px: maximum distance (in pixels) to consider a match an inlier, under squared
                Sampson distance.
        """
        self._use_intrinsics_in_verification = use_intrinsics_in_verification
        self._estimation_threshold_px = estimation_threshold_px
        self._min_matches = (
            NUM_MATCHES_REQ_E_MATRIX if self._use_intrinsics_in_verification else NUM_MATCHES_REQ_F_MATRIX
        )
        # represents i2Ri1=None, i2Ui1=None, v_corr_idxs is an empty array, and inlier_ratio_est_model is 0.0
        self._failure_result = (None, None, np.array([], dtype=np.uint64), 0.0)

    @abc.abstractmethod
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
