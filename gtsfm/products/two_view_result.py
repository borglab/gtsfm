"""Data structure for two-view estimation results.

Authors: Ayush Baid, John Lambert, Zongyue Liu
"""

import dataclasses
from typing import Optional

import numpy as np
from gtsam import Rot3, Unit3

from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport


@dataclasses.dataclass
class TwoViewResult:
    """Output from two-view estimation containing poses and reports.

    The first three fields (i2Ri1, i2Ui1, v_corr_idxs) represent the final pose estimates
    after Inlier Support Processor (ISP) filtering.

    Args:
        i2Ri1: Estimated relative rotation from i1 to i2 (post-ISP).
        i2Ui1: Estimated relative unit translation from i1 to i2 (post-ISP).
        v_corr_idxs: Verified correspondence indices (post-ISP).
        pre_ba_report: Two-view estimation report before bundle adjustment (optional).
        post_ba_report: Two-view estimation report after bundle adjustment (optional).
        post_isp_report: Two-view estimation report after inlier support processing (optional).
        relative_pose_prior: Relative pose prior used in optimization (optional).
        putative_corr_idxs: Putative correspondence indices used for estimation (optional).
    """

    i2Ri1: Optional[Rot3]
    i2Ui1: Optional[Unit3]
    v_corr_idxs: np.ndarray
    pre_ba_report: Optional[TwoViewEstimationReport]
    post_ba_report: Optional[TwoViewEstimationReport]
    post_isp_report: Optional[TwoViewEstimationReport]
    relative_pose_prior: Optional[PosePrior] = None
    putative_corr_idxs: Optional[np.ndarray] = None

    def valid(self) -> bool:
        """Check if both i2Ri1 and i2Ui1 are not None."""
        return self.i2Ri1 is not None and self.i2Ui1 is not None
