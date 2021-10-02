"""Estimator which operates on a pair of images to compute relative pose and verified indices.

Authors: Ayush Baid, John Lambert
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Pose3, Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.verifier.homography import RansacHomographyEstimator
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

import gtsfm.utils.homography_decomposition as homography_decomposition

logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

# In case an epipolar geometry can be verified, it is checked whether
# the geometry describes a planar scene or panoramic view (pure rotation)
# described by a homography. This is a degenerate case, since epipolar
# geometry is only defined for a moving camera. If the inlier ratio of
# a homography comes close to the inlier ratio of the epipolar geometry,
# a planar or panoramic configuration is assumed.
# Based on COLMAP's front-end logic here:
#    https://github.com/colmap/colmap/blob/dev/src/estimators/two_view_geometry.cc#L230
MAX_H_INLIER_RATIO = 0.8

EPSILON = 1e-6


@dataclass(frozen=False)
class TwoViewEstimationReport:
    """Information about verifier result on an edge between two nodes (i1,i2).

    In the spirit of COLMAP's Report class:
    https://github.com/colmap/colmap/blob/dev/src/optim/ransac.h#L82

    Inlier ratio is defined in Heinly12eccv: https://www.cs.unc.edu/~jheinly/publications/eccv2012-heinly.pdf
    or in Slide 59: https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2014/slides/CS4495-Ransac.pdf

    Args:
        v_corr_idxs: verified correspondence indices.
        num_inliers_est_model: #correspondences consistent with estimated model (not necessarily "correct")
        inlier_ratio_est_model: #matches consistent with est. model / # putative matches, i.e.
           measures how consistent the model is with the putative matches.
        num_inliers_gt_model: measures how well the verification worked, w.r.t. GT, i.e. #correct correspondences.
        inlier_ratio_gt_model: #correct matches/#putative matches. Only defined if GT relative pose provided.
        R_error_deg: relative pose error w.r.t. GT. Only defined if GT poses provided.
        U_error_deg: relative translation error w.r.t. GT. Only defined if GT poses provided.
        i2Ri1: relative rotation.
        i2Ui1: relative translation direction.
    """

    v_corr_idxs: np.ndarray
    num_H_inliers: int
    H_inlier_ratio: float
    num_inliers_est_model: float
    inlier_ratio_est_model: Optional[float] = None  # TODO: make not optional (pass from verifier)
    num_inliers_gt_model: Optional[float] = None
    inlier_ratio_gt_model: Optional[float] = None
    R_error_deg: Optional[float] = None
    U_error_deg: Optional[float] = None
    i2Ri1: Optional[Rot3] = None
    i2Ui1: Optional[Unit3] = None


class TwoViewEstimator:
    """Wrapper for running two-view relative pose estimation on image pairs in the dataset."""

    def __init__(
        self,
        matcher: MatcherBase,
        verifier: VerifierBase,
        eval_threshold_px: float,
        min_num_inliers_acceptance: int,
    ) -> None:
        """Initializes the two-view estimator from matcher and verifier.

        Args:
            matcher: matcher to use.
            verifier: verifier to use.
            eval_threshold_px: distance threshold for marking a correspondence pair as inlier during evaluation
                (not during estimation).
            min_num_inliers_acceptance: minimum number of inliers that must agree w/ estimated model, to use
                image pair.
        """
        self._matcher = matcher
        self._verifier = verifier
        self._corr_metric_dist_threshold = eval_threshold_px
        self._min_num_inliers_acceptance = min_num_inliers_acceptance
        # Note: homography estimation threshold must match the E / F thresholds for #inliers to be comparable
        self._homography_estimator = RansacHomographyEstimator(verifier._estimation_threshold_px)

    def get_corr_metric_dist_threshold(self) -> float:
        """Getter for the distance threshold used in the metric for correct correspondences."""
        return self._corr_metric_dist_threshold

    def create_computation_graph(
        self,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        descriptors_i1_graph: Delayed,
        descriptors_i2_graph: Delayed,
        camera_intrinsics_i1_graph: Delayed,
        camera_intrinsics_i2_graph: Delayed,
        im_shape_i1_graph: Delayed,
        im_shape_i2_graph: Delayed,
        i2Ti1_expected_graph: Optional[Delayed] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, Optional[Delayed], Optional[Delayed], Optional[Delayed]]:
        """Create delayed tasks for matching and verification.

        Args:
            keypoints_i1_graph: keypoints for image i1.
            keypoints_i2_graph: keypoints for image i2.
            descriptors_i1_graph: corr. descriptors for image i1.
            descriptors_i2_graph: corr. descriptors for image i2.
            camera_intrinsics_i1_graph: intrinsics for camera i1.
            camera_intrinsics_i2_graph: intrinsics for camera i2.
            im_shape_i1_graph: image shape for image i1.
            im_shape_i2_graph: image shape for image i2.
            i2Ti1_expected_graph (optional): ground truth relative pose, used for evaluation if available. Defaults to
                                             None.

        Returns:
            Computed relative rotation wrapped as Delayed.
            Computed relative translation direction wrapped as Delayed.
            Indices of verified correspondences wrapped as Delayed.
            Error in relative rotation wrapped as Delayed
            Error in relative translation direction wrapped as Delayed.
            Correspondence correctness metrics wrapped as Delayed.
        """

        # graph for matching to obtain putative correspondences
        corr_idxs_graph = self._matcher.create_computation_graph(
            keypoints_i1_graph,
            keypoints_i2_graph,
            descriptors_i1_graph,
            descriptors_i2_graph,
            im_shape_i1_graph,
            im_shape_i2_graph,
        )

        # verification on putative correspondences to obtain relative pose
        # and verified correspondences
        (i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph, inlier_ratio_est_model) = self._verifier.create_computation_graph(
            keypoints_i1_graph,
            keypoints_i2_graph,
            corr_idxs_graph,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
        )

        H_graph, num_H_inliers, H_inlier_ratio, H_inlier_idxs = dask.delayed(
            self._homography_estimator.estimate, nout=4
        )(
            keypoints_i1_graph,
            keypoints_i2_graph,
            match_indices=corr_idxs_graph,
        )

        two_view_report_graph = dask.delayed(generate_two_view_report)(
            inlier_ratio_est_model,
            v_corr_idxs_graph,
            num_H_inliers,
            H_inlier_ratio,
        )

        i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph, two_view_report_graph = dask.delayed(check_for_degeneracy, nout=4)(
            self._min_num_inliers_acceptance,
            H_graph,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
            two_view_report_graph,
            i2Ri1_graph,
            i2Ui1_graph,
            v_corr_idxs_graph,
            keypoints_i1_graph,
            keypoints_i2_graph,
            corr_idxs_graph,
            H_inlier_idxs,
        )

        # if we have the expected GT data, evaluate the computed relative pose
        if i2Ti1_expected_graph is not None:
            two_view_report_graph = dask.delayed(self.add_metrics_wrt_gt_to_report)(
                two_view_report_graph,
                i2Ri1_graph,
                i2Ui1_graph,
                keypoints_i1_graph,
                keypoints_i2_graph,
                camera_intrinsics_i1_graph,
                camera_intrinsics_i2_graph,
                v_corr_idxs_graph,
                i2Ti1_expected_graph,
            )

        return (i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph, two_view_report_graph)

    def add_metrics_wrt_gt_to_report(
        self,
        two_view_report: TwoViewEstimationReport,
        i2Ri1: Rot3,
        i2Ui1: Unit3,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
        v_corr_idxs,
        i2Ti1_expected: Optional[Pose3],
    ) -> TwoViewEstimationReport:
        """ """

        R_error_deg, U_error_deg = compute_relative_pose_metrics(i2Ri1, i2Ui1, i2Ti1_expected)
        num_inliers_gt_model, inlier_ratio_gt_model = compute_correspondence_metrics(
            keypoints_i1,
            keypoints_i2,
            v_corr_idxs,
            camera_intrinsics_i1,
            camera_intrinsics_i2,
            i2Ti1_expected,
            self._corr_metric_dist_threshold,
        )

        two_view_report.R_error_deg = R_error_deg
        two_view_report.U_error_deg = U_error_deg
        two_view_report.num_inliers_gt_model = num_inliers_gt_model
        two_view_report.inlier_ratio_gt_model = inlier_ratio_gt_model

        return two_view_report


def check_for_degeneracy(
    min_num_inliers_acceptance: int,
    H: np.ndarray,
    camera_intrinsics_i1: Cal3Bundler,
    camera_intrinsics_i2: Cal3Bundler,
    two_view_report: TwoViewEstimationReport,
    i2Ri1: Optional[Rot3],
    i2Ui1: Optional[Unit3],
    v_corr_idxs: np.ndarray,
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    corr_idxs: np.ndarray,
    H_inlier_idxs: np.ndarray,
) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
    """GRIC-based multiple model checking.

    http://cmp.felk.cvut.cz/cmp/events/_colloquia/colloquium-2002-04-04/torr.pdf
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.34.3878&rep=rep1&type=pdf

    See https://github.com/colmap/colmap/blob/dev/src/estimators/two_view_geometry.cc#L230

    Args:
        min_num_inliers_acceptance: the minimum number of inliers required to accept image pair.
        H: array of shape (3,3) representing homography matrix.
        camera_intrinsics_i1: camera intrinsics for camera i1.
        camera_intrinsics_i2: camera intrinsics for camera i2.
        two_view_report:
        i2Ri1: estimated relative rotation.
        i2Ui1: estimated relative translation direction.
        v_corr_idxs: verified correspondence indices, as reported by the estimated E or F matrix.
        keypoints_i1:
        keypoints_i2:
        corr_idxs: keypoint matches.
        H_inlier_idxs:
    """
    insufficient_inliers = two_view_report.num_inliers_est_model < min_num_inliers_acceptance
    valid_model = two_view_report.num_inliers_est_model > 0

    if not valid_model or insufficient_inliers:
        # TODO(johnwlambert): also try fitting a homography immediately and check for minimum number of inliers to H.
        logger.info(
            "Degenerate: insufficient number of inliers %d < %d.",
            two_view_report.num_inliers_est_model,
            min_num_inliers_acceptance,
        )
        i2Ri1 = None
        i2Ui1 = None
        v_corr_idxs = np.array([], dtype=np.uint64)
        # remove mention of errors in the report, as pair will be discarded
        two_view_report.R_error_deg = None
        two_view_report.U_error_deg = None
        return i2Ri1, i2Ui1, v_corr_idxs, two_view_report

    H_EF_inlier_ratio = two_view_report.num_H_inliers / (two_view_report.num_inliers_est_model + EPSILON)
    is_planar_or_panoramic = H_EF_inlier_ratio > MAX_H_INLIER_RATIO
    logger.info("H_EF_inlier_ratio: %.2f", H_EF_inlier_ratio)

    if is_planar_or_panoramic:
        logger.info("Planar or panoramic; pose will be extracted from decomposed homography.")
        h_corr_idxs = corr_idxs[H_inlier_idxs]
        # TODO(johnwlambert): count number of inliers for homography, and check against threshold.
        logger.info("Homography had %d inliers.", len(h_corr_idxs))

        # discard normal vector and 3d triangulated points
        i2Ri1, i2Ui1, _, _ = homography_decomposition.pose_from_homography_matrix(
            H=H,
            camera_intrinsics_i1=camera_intrinsics_i1,
            camera_intrinsics_i2=camera_intrinsics_i2,
            points1=keypoints_i1.coordinates[h_corr_idxs[:, 0]],
            points2=keypoints_i2.coordinates[h_corr_idxs[:, 1]],
        )
        two_view_report.v_corr_idxs = h_corr_idxs
        v_corr_idxs = h_corr_idxs

    two_view_report.i2Ri1 = i2Ri1
    two_view_report.i2Ui1 = i2Ui1

    return i2Ri1, i2Ui1, v_corr_idxs, two_view_report


def generate_two_view_report(
    inlier_ratio_est_model: float,
    v_corr_idxs: np.ndarray,
    num_H_inliers: int,
    H_inlier_ratio: float,
) -> TwoViewEstimationReport:
    """Wrapper around class constructor for Dask.

    Note: the following 4 fields are initially set to None, and then updated later if GT is available.
        R_error_deg
        U_error_deg
        num_inliers_gt_model
        inlier_ratio_gt_model
    """
    two_view_report = TwoViewEstimationReport(
        inlier_ratio_est_model=inlier_ratio_est_model,
        num_inliers_est_model=v_corr_idxs.shape[0],
        v_corr_idxs=v_corr_idxs,
        num_H_inliers=num_H_inliers,
        H_inlier_ratio=H_inlier_ratio,
    )
    return two_view_report


def compute_correspondence_metrics(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    corr_idxs_i1i2: np.ndarray,
    intrinsics_i1: Cal3Bundler,
    intrinsics_i2: Cal3Bundler,
    i2Ti1: Pose3,
    epipolar_distance_threshold: float,
) -> Tuple[int, float]:
    """Compute the metrics for the generated verified correspondence.

    Args:
        keypoints_i1: detected keypoints in image i1.
        keypoints_i2: detected keypoints in image i2.
        corr_idxs_i1i2: indices of correspondences.
        intrinsics_i1: intrinsics for i1.
        intrinsics_i2: intrinsics for i2.
        i2Ti1: relative pose.
        epipolar_distance_threshold: max epipolar distance to qualify as a correct match.

    Returns:
        Number of inlier correspondences to ground truth epipolar geometry, i.e. #correct correspondences.
        Inlier Ratio, i.e. ratio of correspondences which are correct w.r.t. given relative pose.
    """
    if corr_idxs_i1i2.size == 0:
        return 0, float("Nan")

    num_inliers_gt_model = metric_utils.count_correct_correspondences(
        keypoints_i1.extract_indices(corr_idxs_i1i2[:, 0]),
        keypoints_i2.extract_indices(corr_idxs_i1i2[:, 1]),
        intrinsics_i1,
        intrinsics_i2,
        i2Ti1,
        epipolar_distance_threshold,
    )
    inlier_ratio_gt_model = num_inliers_gt_model / corr_idxs_i1i2.shape[0]
    return num_inliers_gt_model, inlier_ratio_gt_model


def compute_relative_pose_metrics(
    i2Ri1_computed: Optional[Rot3], i2Ui1_computed: Optional[Unit3], i2Ti1_expected: Pose3
) -> Tuple[Optional[float], Optional[float]]:
    """Compute the metrics on relative camera pose.

    Args:
        i2Ri1_computed: computed relative rotation.
        i2Ui1_computed: computed relative translation direction.
        i2Ti1_expected: expected relative pose.

    Returns:
        Rotation error, in degrees
        Unit translation error, in degrees
    """
    R_error_deg = comp_utils.compute_relative_rotation_angle(i2Ri1_computed, i2Ti1_expected.rotation())
    U_error_deg = comp_utils.compute_relative_unit_translation_angle(
        i2Ui1_computed, Unit3(i2Ti1_expected.translation())
    )

    return (R_error_deg, U_error_deg)


def aggregate_frontend_metrics(
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    angular_err_threshold_deg: float,
    metric_group_name: str,
) -> None:
    """Aggregate the front-end metrics to log summary statistics.

    We define "pose error" as the maximum of the angular errors in rotation and translation, per:
        SuperGlue, CVPR 2020: https://arxiv.org/pdf/1911.11763.pdf
        Learning to find good correspondences. CVPR 2018:
        OA-Net, ICCV 2019:
        NG-RANSAC, ICCV 2019:

    Args:
        two_view_report_dict: report containing front-end metrics for each image pair.
        angular_err_threshold_deg: threshold for classifying angular error metrics as success.
        metric_group_name: name we will assign to the GtsfmMetricGroup returned by this fn.
    """
    num_image_pairs = len(two_view_reports_dict.keys())

    # all rotational errors in degrees
    rot3_angular_errors = []
    trans_angular_errors = []

    inlier_ratio_gt_model_all_pairs = []
    inlier_ratio_est_model_all_pairs = []
    num_inliers_gt_model_all_pairs = []
    num_inliers_est_model_all_pairs = []
    # populate the distributions
    for report in two_view_reports_dict.values():
        rot3_angular_errors.append(report.R_error_deg)
        trans_angular_errors.append(report.U_error_deg)

        inlier_ratio_gt_model_all_pairs.append(report.inlier_ratio_gt_model)
        inlier_ratio_est_model_all_pairs.append(report.inlier_ratio_est_model)
        num_inliers_gt_model_all_pairs.append(report.num_inliers_gt_model)
        num_inliers_est_model_all_pairs.append(report.num_inliers_est_model)

    rot3_angular_errors = np.array(rot3_angular_errors, dtype=float)
    trans_angular_errors = np.array(trans_angular_errors, dtype=float)
    # count number of rot3 errors which are not None. Should be same in rot3/unit3
    num_valid_image_pairs = np.count_nonzero(~np.isnan(rot3_angular_errors))

    # compute pose errors by picking the max error from rot3 and unit3 errors
    pose_errors = np.maximum(rot3_angular_errors, trans_angular_errors)

    # check errors against the threshold
    success_count_rot3 = np.sum(rot3_angular_errors < angular_err_threshold_deg)
    success_count_unit3 = np.sum(trans_angular_errors < angular_err_threshold_deg)
    success_count_pose = np.sum(pose_errors < angular_err_threshold_deg)

    # count image pair entries where inlier ratio w.r.t. GT model == 1.
    all_correct = np.count_nonzero([report.inlier_ratio_gt_model == 1.0 for report in two_view_reports_dict.values()])

    logger.debug(
        "[Two view optimizer] [Summary] Rotation success: %d/%d/%d",
        success_count_rot3,
        num_valid_image_pairs,
        num_image_pairs,
    )

    logger.debug(
        "[Two view optimizer] [Summary] Translation success: %d/%d/%d",
        success_count_unit3,
        num_valid_image_pairs,
        num_image_pairs,
    )

    logger.debug(
        "[Two view optimizer] [Summary] Pose success: %d/%d/%d",
        success_count_pose,
        num_valid_image_pairs,
        num_image_pairs,
    )

    logger.debug(
        "[Two view optimizer] [Summary] # Image pairs with 100%% inlier ratio:: %d/%d", all_correct, num_image_pairs
    )

    # TODO(akshay-krishnan): Move angular_err_threshold_deg and num_total_image_pairs to metadata.
    frontend_metrics = GtsfmMetricsGroup(
        metric_group_name,
        [
            GtsfmMetric("angular_err_threshold_deg", angular_err_threshold_deg),
            GtsfmMetric("num_total_image_pairs", int(num_image_pairs)),
            GtsfmMetric("num_valid_image_pairs", int(num_valid_image_pairs)),
            GtsfmMetric("rotation_success_count", int(success_count_rot3)),
            GtsfmMetric("translation_success_count", int(success_count_unit3)),
            GtsfmMetric("pose_success_count", int(success_count_pose)),
            GtsfmMetric("num_all_inlier_correspondences_wrt_gt_model", int(all_correct)),
            GtsfmMetric("rot3_angular_errors_deg", rot3_angular_errors),
            GtsfmMetric("trans_angular_errors_deg", trans_angular_errors),
            GtsfmMetric("pose_errors_deg", pose_errors),
            GtsfmMetric("inlier_ratio_wrt_gt_model", inlier_ratio_gt_model_all_pairs),
            GtsfmMetric("inlier_ratio_wrt_est_model", inlier_ratio_est_model_all_pairs),
            GtsfmMetric("num_inliers_est_model", num_inliers_est_model_all_pairs),
            GtsfmMetric("num_inliers_gt_model", num_inliers_gt_model_all_pairs),
        ],
    )
    return frontend_metrics
