"""Estimator which operates on a pair of images to compute relative pose and verified indices.

Authors: Ayush Baid, John Lambert
"""
import logging
from typing import Dict, Optional, Tuple

import dask
import numpy as np
import cv2
import trimesh
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Pose3, Rot3, Unit3, PinholeCameraCal3Bundler

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup


logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

EPSILON = 1e-6


class TwoViewEstimator:
    """Wrapper for running two-view relative pose estimation on image pairs in the dataset."""

    def __init__(
        self,
        matcher: MatcherBase,
        verifier: VerifierBase,
        inlier_support_processor: InlierSupportProcessor,
        eval_threshold_px: float,
    ) -> None:
        """Initializes the two-view estimator from matcher and verifier.

        Args:
            matcher: matcher to use.
            verifier: verifier to use.
            inlier_support_processor: post-processor that uses information about RANSAC support to filter out pairs.
            eval_threshold_px: distance threshold for marking a correspondence pair as inlier during evaluation
                (not during estimation).
        """
        self._matcher = matcher
        self._verifier = verifier
        self.processor = inlier_support_processor
        self._corr_metric_dist_threshold = eval_threshold_px

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
        gt_wTi1_graph: Optional[Delayed] = None,
        gt_wTi2_graph: Optional[Delayed] = None,
        gt_scene_mesh_graph: Optional[Delayed] = None,
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
            Two view report w/ verifier metrics wrapped as Delayed.
            Two view report w/ post-processor metrics wrapped as Delayed.
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
        # TODO: name this verified_correspondence_idxs (add note: everything here is delayed)
        (i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph, inlier_ratio_est_model) = self._verifier.create_computation_graph(
            keypoints_i1_graph,
            keypoints_i2_graph,
            corr_idxs_graph,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
        )
        # gt_verifier_result = dask.delayed(ground_truth_verifier)(
        #     keypoints_i1_graph,
        #     keypoints_i2_graph,
        #     corr_idxs_graph,
        #     camera_intrinsics_i1_graph,
        #     camera_intrinsics_i2_graph,
        #     self._corr_metric_dist_threshold,
        #     gt_wTi1_graph,
        #     gt_wTi2_graph,
        #     gt_scene_mesh_graph,
        # )
        # i2Ri1_graph = gt_verifier_result[0]
        # i2Ui1_graph = gt_verifier_result[1]
        # v_corr_idxs_graph = gt_verifier_result[2]
        # inlier_ratio_est_model = gt_verifier_result[3]

        # if we have the expected GT data, evaluate the computed relative pose
        if gt_wTi1_graph is not None and gt_wTi2_graph is not None:
            i2Ti1_expected_graph = gt_wTi2_graph.between(gt_wTi1_graph)
            R_error_deg, U_error_deg = dask.delayed(compute_relative_pose_metrics, nout=2)(
                i2Ri1_graph, i2Ui1_graph, i2Ti1_expected_graph
            )
            v_corr_idxs_inlier_mask_gt, reproj_error_gt_model = dask.delayed(compute_correspondence_metrics, nout=2)(
                keypoints_i1_graph,
                keypoints_i2_graph,
                v_corr_idxs_graph,
                camera_intrinsics_i1_graph,
                camera_intrinsics_i2_graph,
                self._corr_metric_dist_threshold,
                gt_wTi1_graph,
                gt_wTi2_graph,
                gt_scene_mesh_graph,
            )
        else:
            R_error_deg, U_error_deg = None, None
            v_corr_idxs_inlier_mask_gt, reproj_error_gt_model = None, None

        two_view_report_graph = dask.delayed(generate_two_view_report)(
            inlier_ratio_est_model,
            v_corr_idxs_graph,
            R_error_deg=R_error_deg,
            U_error_deg=U_error_deg,
            v_corr_idxs_inlier_mask_gt=v_corr_idxs_inlier_mask_gt,
            reproj_error_gt_model=reproj_error_gt_model,
        )

        # Note: We name the output as _pp, as it represents a post-processed quantity.
        (
            i2Ri1_pp_graph,
            i2Ui1_pp_graph,
            v_corr_idxs_pp_graph,
            two_view_report_pp_graph,
        ) = self.processor.create_computation_graph(i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph, two_view_report_graph)
        # We provide both, as we will create reports for both.
        return (i2Ri1_pp_graph, i2Ui1_pp_graph, v_corr_idxs_pp_graph, two_view_report_graph, two_view_report_pp_graph)


def ground_truth_verifier(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    matches_i1i2: np.ndarray,
    intrinsics_i1: Cal3Bundler,
    intrinsics_i2: Cal3Bundler,
    dist_threshold: float,
    gt_wTi1: Pose3,
    gt_wTi2: Pose3,
    gt_scene_mesh: trimesh.Trimesh,
) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, float]:
    """Verify correspondences and compute relative pose using ground truth data."""
    logger.info("ground truth verifier")
    # Compute ground truth correspondences.
    gt_camera_i1 = PinholeCameraCal3Bundler(gt_wTi1, intrinsics_i1)
    gt_camera_i2 = PinholeCameraCal3Bundler(gt_wTi2, intrinsics_i2)
    inlier_mask, _ = metric_utils.mesh_inlier_correspondences(
        keypoints_i1.extract_indices(matches_i1i2[:, 0]),
        keypoints_i2.extract_indices(matches_i1i2[:, 1]),
        gt_camera_i1,
        gt_camera_i2,
        gt_scene_mesh,
        dist_threshold,
    )
    inlier_idxs = np.where(inlier_mask.ravel() == 1)[0]
    v_corr_idxs = matches_i1i2[inlier_idxs]
    logger.info("computed ground truth correspondences")

    if v_corr_idxs.shape[0] < 15:
        return None, None, np.array([], dtype=np.uint64), 0.0

    # Compute essential matrix.
    i2Ei1, _ = cv2.findEssentialMat(
        keypoints_i1.extract_indices(v_corr_idxs[:, 0]).coordinates,
        keypoints_i2.extract_indices(v_corr_idxs[:, 1]).coordinates,
        intrinsics_i1.K(),
        method=0,
    )
    logger.info("computed ground truth essential matrix")

    (i2Ri1, i2Ui1) = verification_utils.recover_relative_pose_from_essential_matrix(
        i2Ei1,
        keypoints_i1.coordinates[v_corr_idxs[:, 0]],
        keypoints_i2.coordinates[v_corr_idxs[:, 1]],
        intrinsics_i1,
        intrinsics_i2,
    )

    return i2Ri1, i2Ui1, v_corr_idxs, 1.0


def generate_two_view_report(
    inlier_ratio_est_model: float,
    v_corr_idxs: np.ndarray,
    R_error_deg: Optional[float] = None,
    U_error_deg: Optional[float] = None,
    v_corr_idxs_inlier_mask_gt: Optional[np.ndarray] = None,
    reproj_error_gt_model: Optional[np.ndarray] = None,
) -> TwoViewEstimationReport:
    """Wrapper around class constructor for Dask."""
    # Compute ground truth metrics.
    if v_corr_idxs_inlier_mask_gt is not None and reproj_error_gt_model is not None:
        num_inliers_gt_model = np.count_nonzero(v_corr_idxs_inlier_mask_gt)
        inlier_ratio_gt_model = (
            np.count_nonzero(v_corr_idxs_inlier_mask_gt) / v_corr_idxs.shape[0] if len(v_corr_idxs) > 0 else 0.0
        )
        inlier_avg_reproj_error_gt_model = np.mean(reproj_error_gt_model[v_corr_idxs_inlier_mask_gt])
        outlier_avg_reproj_error_gt_model = np.nanmean(
            reproj_error_gt_model[np.logical_not(v_corr_idxs_inlier_mask_gt)]
        )
    else:
        num_inliers_gt_model = 0
        inlier_ratio_gt_model = float("Nan")
        inlier_avg_reproj_error_gt_model = float("Nan")
        outlier_avg_reproj_error_gt_model = float("Nan")

    # Generate report.
    two_view_report = TwoViewEstimationReport(
        inlier_ratio_est_model=inlier_ratio_est_model,
        num_inliers_est_model=v_corr_idxs.shape[0],
        num_inliers_gt_model=num_inliers_gt_model,
        inlier_ratio_gt_model=inlier_ratio_gt_model,
        v_corr_idxs_inlier_mask_gt=v_corr_idxs_inlier_mask_gt,
        v_corr_idxs=v_corr_idxs,
        R_error_deg=R_error_deg,
        U_error_deg=U_error_deg,
        reproj_error_gt_model=reproj_error_gt_model,
        inlier_avg_reproj_error_gt_model=inlier_avg_reproj_error_gt_model,
        outlier_avg_reproj_error_gt_model=outlier_avg_reproj_error_gt_model,
    )
    return two_view_report


def compute_correspondence_metrics(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    corr_idxs_i1i2: np.ndarray,
    intrinsics_i1: Cal3Bundler,
    intrinsics_i2: Cal3Bundler,
    epipolar_distance_threshold: float,
    wTi1: Optional[Pose3] = None,
    wTi2: Optional[Pose3] = None,
    gt_scene_mesh: Optional[trimesh.Trimesh] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
        Mask of which verified correspondences are classified as correct under Sampson error
            (using GT epipolar geometry).
    """
    if corr_idxs_i1i2.size == 0:
        return None, None

    v_corr_idxs_inlier_mask_gt, reproj_error_gt_model = metric_utils.count_correct_correspondences(
        keypoints_i1.extract_indices(corr_idxs_i1i2[:, 0]),
        keypoints_i2.extract_indices(corr_idxs_i1i2[:, 1]),
        intrinsics_i1,
        intrinsics_i2,
        epipolar_distance_threshold,
        wTi1,
        wTi2,
        gt_scene_mesh,
    )
    return v_corr_idxs_inlier_mask_gt, reproj_error_gt_model


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
    two_view_reports_dict: Dict[Tuple[int, int], Optional[TwoViewEstimationReport]],
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
        if report is None:
            continue
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
    all_correct = np.count_nonzero(
        [report.inlier_ratio_gt_model == 1.0 for report in two_view_reports_dict.values() if report is not None]
    )

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
