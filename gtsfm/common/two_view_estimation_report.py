"""Container for results about a image pair.

Authors: John Lambert
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from gtsam import Rot3, Unit3


class TwoViewConfigurationType(str, Enum):
    """Categorization of the configuration of two views.

    Calibrated indicates that the two-view relation has sufficient parallax, and intrinsics are known and reliable.
    Uncalibrated indicates that the two-view relation has sufficient parallax, but intrinsics are unknown / unreliable.
    Planar / Panoramic indicates that the relation is a homography.
    Degenerate indicates that there was insufficient support to accurately determine
        the two-view relation.
    """

    CALIBRATED: str = "CALIBRATED"
    UNCALIBRATED: str = "UNCALIBRATED"
    PLANAR_OR_PANORAMIC: str = "PLANAR_OR_PANORAMIC"
    DEGENERATE: str = "DEGENERATE"
    UNKNOWN: str = "UNKNOWN"


@dataclass(frozen=False)
class TwoViewEstimationReport:
    """Information about verifier result on an edge between two nodes (i1,i2).

    In the spirit of COLMAP's Report class:
    https://github.com/colmap/colmap/blob/dev/src/optim/ransac.h#L82

    Inlier ratio is defined in Heinly12eccv: https://www.cs.unc.edu/~jheinly/publications/eccv2012-heinly.pdf
    or in Slide 59: https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2014/slides/CS4495-Ransac.pdf

    Args:
        configuration_type: classification of two-view relationship.
        v_corr_idxs: verified correspondence indices.
        num_inliers_H: number of #correspondences consistent with estimated homography model.
        inlier_ratio_H: fraction representating #correspondences consistent with homography / (# putatives matches).
        num_inliers_est_model: #correspondences consistent with estimated model (not necessarily "correct")
        inlier_ratio_est_model: Fraction representing (#matches consistent with est. model) / (# putative matches),
            i.e. measures how consistent the model is with the putative matches.
        num_inliers_gt_model: measures how well the verification worked, w.r.t. GT, i.e. #correct correspondences.
        inlier_ratio_gt_model: #correct matches/#putative matches. Only defined if GT relative pose provided.
        v_corr_idxs_inlier_mask_gt: Mask of which verified correspondences are classified as correct under
            Sampson error (using GT epipolar geometry).
        R_error_deg: relative pose error w.r.t. GT. Only defined if GT poses provided.
        U_error_deg: relative translation error w.r.t. GT. Only defined if GT poses provided.
        reproj_errors_gt_model: reprojection errors between correspondences w.r.t. GT.
    """

    configuration_type: TwoViewConfigurationType
    v_corr_idxs: np.ndarray
    num_inliers_H: int
    inlier_ratio_H: float
    num_inliers_est_model: float
    inlier_ratio_est_model: Optional[float] = None  # TODO: make not optional (pass from verifier)
    num_inliers_gt_model: Optional[float] = None
    inlier_ratio_gt_model: Optional[float] = None
    v_corr_idxs_inlier_mask_gt: Optional[np.ndarray] = None
    R_error_deg: Optional[float] = None
    U_error_deg: Optional[float] = None
    reproj_errors_gt_model: Optional[np.ndarray] = None
    