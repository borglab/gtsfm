"""Estimator which operates on a pair of images to compute relative pose and verified indices.

Authors: Ayush Baid, John Lambert
"""
import logging
from typing import Dict, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Pose3, PinholeCameraCal3Bundler, Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.data_association.point3d_initializer import TriangulationParam
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup


logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


DATA_ASSOC_REPROJ_ERROR_THRESH = 3
DATA_ASSOC_2VIEW = DataAssociation(
    reproj_error_thresh=DATA_ASSOC_REPROJ_ERROR_THRESH, min_track_len=2, mode=TriangulationParam.NO_RANSAC
)
BUNDLE_ADJUST_2VIEW = BundleAdjustmentOptimizer(
    output_reproj_error_thresh=100, robust_measurement_noise=True
)  # we dont care about output error threshold as we do not access the tracks of 2-view BA, which the threshold affects.


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

    @classmethod
    def bundle_adjust(
        cls,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        verified_corr_idxes: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
        i2Ri1_initial: Optional[Rot3],
        i2Ui1_initial: Optional[Unit3],
    ) -> Tuple[Optional[Rot3], Optional[Unit3]]:
        """Refine the relative pose using bundle adjustment on the 2-view scene.

        Args:
            keypoints_i1: keypoints from image i1.
            keypoints_i2: keypoints from image i2.
            verified_corr_idxes: indices of verified correspondences between i1 and i2.
            camera_intrinsics_i1: intrinsics for i1.
            camera_intrinsics_i2: intrinsics for i2.
            i2Ri1_initial: the relative rotation to be used as initial rotation between cameras.
            i2Ui1_initial: the relative unit direction, to be used to initialize initial translation between cameras.

        Returns:
            Optimized relative rotation i2Ri1.
            Optimized unit translation i2Ui1.
        """

        if i2Ri1_initial is None or i2Ui1_initial is None:
            return None, None

        # set the camera 2 pose as the global coordinate system
        camera_i1 = PinholeCameraCal3Bundler(Pose3(i2Ri1_initial, i2Ui1_initial.point3()), camera_intrinsics_i1)
        camera_i2 = PinholeCameraCal3Bundler(Pose3(), camera_intrinsics_i2)

        # perform data association to construct 2-view BA input
        ba_input, _ = DATA_ASSOC_2VIEW.run(
            num_images=2,
            cameras={0: camera_i1, 1: camera_i2},
            corr_idxs_dict={(0, 1): verified_corr_idxes},
            keypoints_list=[keypoints_i1, keypoints_i2],
        )

        ba_output, _ = BUNDLE_ADJUST_2VIEW.run(ba_input)

        # extract the camera poses
        wTi1, wTi2 = ba_output.get_camera_poses()

        if wTi1 is None or wTi2 is None:
            logger.warning("2-view BA failed")
            return i2Ri1_initial, i2Ui1_initial

        i2Ti1_optimized = wTi2.between(wTi1)

        return i2Ti1_optimized.rotation(), Unit3(i2Ti1_optimized.translation())

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

        # verification on putative correspondences to obtain relative pose and verified correspondences\
        # TODO: name this verified_correspondence_idxs (add note: everything here is delayed)
        (
            i2Ri1_pre_ba_graph,
            i2Ui1_pre_ba_graph,
            v_corr_idxs_graph,
            inlier_ratio_est_model,
        ) = self._verifier.create_computation_graph(
            keypoints_i1_graph,
            keypoints_i2_graph,
            corr_idxs_graph,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
        )

        i2Ri1_graph, i2Ui1_graph = dask.delayed(self.bundle_adjust, nout=2)(
            keypoints_i1_graph,
            keypoints_i2_graph,
            v_corr_idxs_graph,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
            i2Ri1_pre_ba_graph,
            i2Ui1_pre_ba_graph,
        )

        # if we have the expected GT data, evaluate the computed relative pose
        if i2Ti1_expected_graph is not None:
            R_error_deg, U_error_deg = dask.delayed(compute_relative_pose_metrics, nout=2)(
                i2Ri1_graph, i2Ui1_graph, i2Ti1_expected_graph
            )
            num_inliers_gt_model, inlier_ratio_gt_model, v_corr_idxs_inlier_mask_gt = dask.delayed(
                compute_correspondence_metrics, nout=3
            )(
                keypoints_i1_graph,
                keypoints_i2_graph,
                v_corr_idxs_graph,
                camera_intrinsics_i1_graph,
                camera_intrinsics_i2_graph,
                i2Ti1_expected_graph,
                self._corr_metric_dist_threshold,
            )
        else:
            R_error_deg, U_error_deg = None, None
            num_inliers_gt_model, inlier_ratio_gt_model = None, None
            v_corr_idxs_inlier_mask_gt = None

        two_view_report_graph = dask.delayed(generate_two_view_report)(
            inlier_ratio_est_model,
            R_error_deg,
            U_error_deg,
            num_inliers_gt_model,
            inlier_ratio_gt_model,
            v_corr_idxs_inlier_mask_gt,
            v_corr_idxs_graph,
        )

        # Note: We name the output as _pp, as it represents a post-processed quantity.
        (
            i2Ri1_pp_graph,
            i2Ui1_pp_graph,
            v_corr_idxs_pp_graph,
            two_view_report_pp_graph,
        ) = self.processor.create_computation_graph(
            i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph, two_view_report_graph
        )
        # We provide both, as we will create reports for both.
        return (i2Ri1_pp_graph, i2Ui1_pp_graph, v_corr_idxs_pp_graph, two_view_report_graph, two_view_report_pp_graph)


def generate_two_view_report(
    inlier_ratio_est_model: float,
    R_error_deg: float,
    U_error_deg: float,
    num_inliers_gt_model: int,
    inlier_ratio_gt_model: float,
    v_corr_idxs_inlier_mask_gt: np.ndarray,
    v_corr_idxs: np.ndarray,
) -> TwoViewEstimationReport:
    """Wrapper around class constructor for Dask."""
    two_view_report = TwoViewEstimationReport(
        inlier_ratio_est_model=inlier_ratio_est_model,
        num_inliers_est_model=v_corr_idxs.shape[0],
        num_inliers_gt_model=num_inliers_gt_model,
        inlier_ratio_gt_model=inlier_ratio_gt_model,
        v_corr_idxs_inlier_mask_gt=v_corr_idxs_inlier_mask_gt,
        v_corr_idxs=v_corr_idxs,
        R_error_deg=R_error_deg,
        U_error_deg=U_error_deg,
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
) -> Tuple[int, float, Optional[np.ndarray]]:
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
        return 0, float("Nan"), None

    v_corr_idxs_inlier_mask_gt = metric_utils.count_correct_correspondences(
        keypoints_i1.extract_indices(corr_idxs_i1i2[:, 0]),
        keypoints_i2.extract_indices(corr_idxs_i1i2[:, 1]),
        intrinsics_i1,
        intrinsics_i2,
        i2Ti1,
        epipolar_distance_threshold,
    )
    num_inliers_gt_model = np.count_nonzero(v_corr_idxs_inlier_mask_gt)
    inlier_ratio_gt_model = num_inliers_gt_model / corr_idxs_i1i2.shape[0]
    return num_inliers_gt_model, inlier_ratio_gt_model, v_corr_idxs_inlier_mask_gt


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
