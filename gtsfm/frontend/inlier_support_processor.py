"""Post-processor that uses information about RANSAC support (for verified correspondences) to filter out image pairs.

Authors: John Lambert
"""
import dataclasses
from typing import Optional, Tuple

import numpy as np
from gtsam import Rot3, Unit3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport

logger = logger_utils.get_logger()


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

    def apply(
        self,
        i2Ri1: Optional[Rot3],
        i2Ui1: Optional[Unit3],
        v_corr_idxs: np.ndarray,
        two_view_report: TwoViewEstimationReport,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, Optional[TwoViewEstimationReport]]:
        """Checks for sufficient support among correspondences to estimate a trustworthy relative pose for image pair.

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
        # i2Ri1 is None, and i2Ui1 is None upon failure.
        failure_result = (
            None,
            None,
            np.array([], dtype=np.uint64),
            TwoViewEstimationReport(v_corr_idxs=v_corr_idxs, num_inliers_est_model=0),
        )

        # make a copy of the report
        two_view_report_post_isp = dataclasses.replace(two_view_report)

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
            return failure_result

        if valid_model and insufficient_inliers:
            logger.debug(
                "Insufficient number of inliers. %d vs. %d",
                two_view_report.num_inliers_est_model,
                self._min_num_inliers_est_model,
            )
            return failure_result

        return i2Ri1, i2Ui1, v_corr_idxs, two_view_report_post_isp
