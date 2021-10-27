"""Post-processor that uses information about RANSAC support (for verified correspondences) to filter out image pairs.

Authors: John Lambert
"""

from typing import Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Rot3, Unit3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport, TwoViewConfigurationType


logger = logger_utils.get_logger()

EPSILON = 1e-6

# In case an epipolar geometry can be verified, it is checked whether
# the geometry describes a planar scene or panoramic view (pure rotation)
# described by a homography. This is a degenerate case, since epipolar
# geometry is only defined for a moving camera. If the inlier ratio of
# a homography comes close to the inlier ratio of the epipolar geometry,
# a planar or panoramic configuration is assumed.
# Based on COLMAP's front-end logic here:
#    https://github.com/colmap/colmap/blob/dev/src/estimators/two_view_geometry.cc#L230
MAX_H_INLIER_RATIO = 0.8


class InlierSupportProcessor:
    """Reasons about the amount of support for a relative pose measurement between an image pair."""

    def __init__(
        self,
        min_num_inliers_est_model: int,
        min_inlier_ratio_est_model: float,
    ) -> None:
        """Saves inlier thresholds to use for filtering.

        Args:
            min_num_inliers_est_model: minimum number of inliers that must agree w/ estimated model, to accept
                and use the image pair.
            min_inlier_ratio_est_model: minimum allowed inlier ratio w.r.t. the estimated model to accept
                the verification result and use the image pair, i.e. the lowest allowed ratio of
                #final RANSAC inliers/ #putatives. A lower fraction indicates less agreement among the result.
        """
        self._min_num_inliers_est_model = min_num_inliers_est_model
        self._min_inlier_ratio_est_model = min_inlier_ratio_est_model

    def run(
        self,
        i2Ri1: Optional[Rot3],
        i2Ui1: Optional[Unit3],
        v_corr_idxs: np.ndarray,
        two_view_report: TwoViewEstimationReport,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, Optional[TwoViewEstimationReport]]:
        """Check for sufficient support among correspondences to estimate a trustworthy relative pose for this image pair.

        We don't modify the report (to stay functional), but report InlierSupportProcessor metrics separately.

        Args:
            i2Ri1: relative rotation measurement.
            i2Ui1: relative translation direction measurement:
            v_corr_idxs: verified correspondence indices as (N,2) array.
            two_view_report: two-view estimation report.

        Returns:
            i2Ri1: relative rotation, or None if insufficient support
            i2Ui1: relative translation direction, or None if insufficient support
            v_corr_idxs: empty (0,2) array if insufficient support
            two_view_report: two-view estimation report, or None if insufficient support
        """
        insufficient_inliers = two_view_report.num_inliers_est_model < self._min_num_inliers_est_model

        # TODO: technically this should almost always be non-zero, just need to move up to earlier
        valid_model = two_view_report.num_inliers_est_model > 0

        # no need to extract the relative pose if we have insufficient inliers.
        if two_view_report.inlier_ratio_est_model < self._min_inlier_ratio_est_model:
            logger.debug(
                "Insufficient inlier ratio. %d vs. %d",
                two_view_report.inlier_ratio_est_model,
                self._min_inlier_ratio_est_model,
            )
            i2Ri1 = None
            i2Ui1 = None
            v_corr_idxs = np.array([], dtype=np.uint64)
            two_view_report.configuration_type = TwoViewConfigurationType.DEGENERATE
            return i2Ri1, i2Ui1, v_corr_idxs, two_view_report

        if valid_model and insufficient_inliers:
            logger.debug(
                "Insufficient number of inliers. %d vs. %d",
                two_view_report.num_inliers_est_model,
                self._min_num_inliers_est_model,
            )
            i2Ri1 = None
            i2Ui1 = None
            v_corr_idxs = np.array([], dtype=np.uint64)
            two_view_report.configuration_type = TwoViewConfigurationType.DEGENERATE
            return i2Ri1, i2Ui1, v_corr_idxs, two_view_report

        H_EF_inlier_ratio = two_view_report.num_H_inliers / (two_view_report.num_inliers_est_model + EPSILON)
        is_planar_or_panoramic = H_EF_inlier_ratio > MAX_H_INLIER_RATIO
        logger.info("H_EF_inlier_ratio: %.2f", H_EF_inlier_ratio)
        if is_planar_or_panoramic:
            logger.info("Planar or panoramic; ignoring image pair (in future PR pose will be extracted from decomposed homography).")
            i2Ri1 = None
            i2Ui1 = None
            v_corr_idxs = np.array([], dtype=np.uint64)
            two_view_report.configuration_type = TwoViewConfigurationType.PLANAR_OR_PANORAMIC
            return i2Ri1, i2Ui1, v_corr_idxs, two_view_report

        two_view_report.i2Ri1 = i2Ri1
        two_view_report.i2Ui1 = i2Ui1
        two_view_report.configuration_type = TwoViewConfigurationType.CALIBRATED

        return i2Ri1, i2Ui1, v_corr_idxs, two_view_report

    def create_computation_graph(
        self, i2Ri1_graph: Delayed, i2Ui1_graph: Delayed, v_corr_idxs_graph: Delayed, two_view_report_graph: Delayed
    ) -> Tuple[Delayed, Delayed, Delayed, Delayed]:
        """Create the Dask computational graph for the InlierSupportProcessor.

        Args:
            i2Ri1_graph: Relative rotation i2Ri1 for a single (i1,i2) image pair, wrapped up as Delayed.
                When computed, will be of type Rot3, or may be None.
            i2Ui1_graph: Relative translation direction i2Ui1 for a single (i1,i2) image pair, wrapped up as Delayed.
                When computed, will be of type Unit3, and may be None.
            v_corr_idxs_graph: Keypoint indices for verified correspondences, wrapped up as Delayed.
               When computed, will be an array of shape (N,2).
            two_view_report_graph: Report from verifier indicating the amount of "support" found for the estimated
                relative pose.

        Returns:
            i2Ri1_pp_graph: Post-processed relative rotation (may now be None, if insufficient support).
            i2Ui1_pp_graph: Post-processed relative translation direction (may now be None, if insufficient support).
            v_corr_idxs_pp_graph: Post-processed keypoint indices for verified correspondences.
                May now be an empty array, if insufficient support.
            two_view_report_pp_graph: Post-processed two-view report (may now be None, if insufficient support).
        """
        return dask.delayed(self.run, nout=4)(i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph, two_view_report_graph)
