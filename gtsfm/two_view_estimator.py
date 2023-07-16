"""Estimator which operates on a pair of images to compute relative pose and verified indices.

Authors: Ayush Baid, John Lambert
"""
import dataclasses
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple

from dask.distributed import Client
import numpy as np
from gtsam import PinholeCameraCal3Bundler, Pose3, Rot3, Unit3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
import gtsfm.frontend.two_view_ba_utils as two_view_ba_utils
from gtsfm.bundle.two_view_ba import TwoViewBundleAdjustment
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.data_association.point3d_initializer import TriangulationOptions
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.verifier.verifier_base import VerifierBase

logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


class TWO_VIEW_REPORT_TAG(str, Enum):
    PRE_BA = "PRE_BA_2VIEW_REPORT"
    POST_BA = "POST_BA_2VIEW_REPORT"
    POST_ISP = "POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT"
    VIEWGRAPH = "VIEWGRAPH_2VIEW_REPORT"
    CORRESPONDENCE_AUGMENTED = "CORR_AURGMENTED_2VIEW_REPORT"


TWO_VIEW_OUTPUT = Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, Dict[TWO_VIEW_REPORT_TAG, TwoViewEstimationReport]]


class TwoViewEstimator:
    """Wrapper for running two-view relative pose estimation on image pairs in the dataset."""

    def __init__(
        self,
        verifier: VerifierBase,
        inlier_support_processor: InlierSupportProcessor,
        bundle_adjust_2view: bool,
        eval_threshold_px: float,
        triangulation_options: TriangulationOptions,
        bundle_adjust_2view_maxiters: int = 100,
        ba_reproj_error_thresholds: List[Optional[float]] = [0.5],
    ) -> None:
        """Initializes the two-view estimator from verifier.

        Args:
            verifier: Verifier to use.
            inlier_support_processor: Post-processor that uses information about RANSAC support to filter out pairs.
            bundle_adjust_2view: Boolean flag indicating if bundle adjustment is to be run on the 2-view data.
            eval_threshold_px: Distance threshold for marking a correspondence pair as inlier during evaluation
                (not during estimation).
            bundle_adjust_2view_maxiters (optional): Max number of iterations for 2-view BA. Defaults to 100.
            ba_reproj_error_thresholds (optional): Reprojection thresholds used to filter features after each stage of
                2-view BA. The length of this list decides the number of BA stages. Defaults to [0.5] (single stage).
        """
        self._verifier = verifier
        self.processor = inlier_support_processor
        self._bundle_adjust_2view = bundle_adjust_2view
        self._corr_metric_dist_threshold = eval_threshold_px
        self._triangulation_options = triangulation_options
        self._ba_optimizer = TwoViewBundleAdjustment(
            reproj_error_thresholds=ba_reproj_error_thresholds,
            robust_measurement_noise=True,
            max_iterations=bundle_adjust_2view_maxiters,
        )

    def get_corr_metric_dist_threshold(self) -> float:
        """Getter for the distance threshold used in the metric for correct correspondences."""
        return self._corr_metric_dist_threshold

    def run_2view(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        putative_corr_idxs: np.ndarray,
        camera_intrinsics_i1: Optional[gtsfm_types.CALIBRATION_TYPE],
        camera_intrinsics_i2: Optional[gtsfm_types.CALIBRATION_TYPE],
        i2Ti1_prior: Optional[PosePrior],
        gt_camera_i1: Optional[gtsfm_types.CAMERA_TYPE],
        gt_camera_i2: Optional[gtsfm_types.CAMERA_TYPE],
        gt_scene_mesh: Optional[Any] = None,
    ) -> TWO_VIEW_OUTPUT:
        """Estimate relative pose between two views, using verification."""
        # verification on putative correspondences to obtain relative pose and verified correspondences
        (pre_ba_i2Ri1, pre_ba_i2Ui1, pre_ba_v_corr_idxs, pre_ba_inlier_ratio_wrt_estimate) = self._verifier.verify(
            keypoints_i1,
            keypoints_i2,
            putative_corr_idxs,
            camera_intrinsics_i1,
            camera_intrinsics_i2,
        )

        pre_ba_report = generate_two_view_report_from_result(
            i2Ri1_computed=pre_ba_i2Ri1,
            i2Ui1_computed=pre_ba_i2Ui1,
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            verified_corr_idxs=pre_ba_v_corr_idxs,
            inlier_ratio_wrt_estimate=pre_ba_inlier_ratio_wrt_estimate,
            gt_camera_i1=gt_camera_i1,
            gt_camera_i2=gt_camera_i2,
            gt_scene_mesh=gt_scene_mesh,
            corr_metric_dist_threshold=self.get_corr_metric_dist_threshold(),
        )

        # Optionally, do two-view bundle adjustment
        if self._bundle_adjust_2view and len(pre_ba_v_corr_idxs) >= self.processor._min_num_inliers_est_model:
            post_ba_i2Ri1, post_ba_i2Ui1, post_ba_v_corr_idxs = two_view_ba_utils.bundle_adjust(
                keypoints_i1,
                keypoints_i2,
                pre_ba_v_corr_idxs,
                camera_intrinsics_i1,
                camera_intrinsics_i2,
                pre_ba_i2Ri1,
                pre_ba_i2Ui1,
                i2Ti1_prior,
                triangulation_options=self._triangulation_options,
                ba_optimizer=self._ba_optimizer,
            )
            post_ba_inlier_ratio_wrt_estimate = float(len(post_ba_v_corr_idxs)) / len(putative_corr_idxs)

            # TODO: Remove this hack once we can handle the lower post_ba_inlier_ratio_wrt_estimate downstream.
            post_ba_inlier_ratio_wrt_estimate = pre_ba_inlier_ratio_wrt_estimate

            post_ba_report = generate_two_view_report_from_result(
                i2Ri1_computed=post_ba_i2Ri1,
                i2Ui1_computed=post_ba_i2Ui1,
                keypoints_i1=keypoints_i1,
                keypoints_i2=keypoints_i2,
                verified_corr_idxs=post_ba_v_corr_idxs,
                inlier_ratio_wrt_estimate=post_ba_inlier_ratio_wrt_estimate,
                gt_camera_i1=gt_camera_i1,
                gt_camera_i2=gt_camera_i2,
                gt_scene_mesh=gt_scene_mesh,
                corr_metric_dist_threshold=self.get_corr_metric_dist_threshold(),
            )
        else:
            post_ba_i2Ri1 = pre_ba_i2Ri1
            post_ba_i2Ui1 = pre_ba_i2Ui1
            post_ba_v_corr_idxs = pre_ba_v_corr_idxs
            post_ba_report = dataclasses.replace(pre_ba_report)

        (
            post_isp_i2Ri1,
            post_isp_i2Ui1,
            post_isp_v_corr_idxs,
            post_isp_report,
        ) = self.processor.run_inlier_support(post_ba_i2Ri1, post_ba_i2Ui1, post_ba_v_corr_idxs, post_ba_report)

        two_view_reports = {
            TWO_VIEW_REPORT_TAG.PRE_BA: pre_ba_report,
            TWO_VIEW_REPORT_TAG.POST_BA: post_ba_report,
            TWO_VIEW_REPORT_TAG.POST_ISP: post_isp_report,
        }

        return post_isp_i2Ri1, post_isp_i2Ui1, post_isp_v_corr_idxs, two_view_reports


def generate_two_view_report_from_result(
    i2Ri1_computed: Optional[Rot3],
    i2Ui1_computed: Optional[Unit3],
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    verified_corr_idxs: np.ndarray,
    inlier_ratio_wrt_estimate: float,
    gt_camera_i1: Optional[gtsfm_types.CAMERA_TYPE],
    gt_camera_i2: Optional[gtsfm_types.CAMERA_TYPE],
    gt_scene_mesh: Optional[Any],
    corr_metric_dist_threshold: float,
) -> TwoViewEstimationReport:
    """Generate a TwoViewEstimationReport from the results of the two-view estimation.

    Currently metrics wrt GT camera only supports cases where gt_camera is PinholeCameraCal3Bundler.

    Args:
        i2Ri1_computed: Computed relative rotation.
        i2Ui1_computed: Computed relative unit translation.
        keypoints_i1: Keypoints from image i1.
        keypoints_i2: Keypoints from image i2.
        verified_corr_idxs: Indices of verified correspondences between i1 and i2.
        inlier_ratio_wrt_estimate: Inlier ratio w.r.t. the estimated relative pose.
        gt_camera_i1: Ground truth camera for i1.
        gt_camera_i2: Ground truth camera for i2.
        gt_scene_mesh: Ground truth scene mesh.

    Returns:
        TwoViewEstimationReport object, some fields may be None if either gt_camera are None.
    """
    if gt_camera_i1 and gt_camera_i2:
        # if we have the expected GT data, evaluate the computed relative pose
        R_error_deg, U_error_deg = compute_relative_pose_metrics(
            i2Ri1_computed, i2Ui1_computed, gt_camera_i1.pose(), gt_camera_i2.pose()
        )
        # TODO: add support for other camera models
        if isinstance(gt_camera_i1, PinholeCameraCal3Bundler) and isinstance(gt_camera_i2, PinholeCameraCal3Bundler):
            inlier_mask_wrt_gt, reproj_error_wrt_gt = metric_utils.compute_correspondence_metrics(
                keypoints_i1=keypoints_i1,
                keypoints_i2=keypoints_i2,
                corr_idxs_i1i2=verified_corr_idxs,
                dist_threshold=corr_metric_dist_threshold,
                gt_camera_i1=gt_camera_i1,
                gt_camera_i2=gt_camera_i2,
                gt_scene_mesh=gt_scene_mesh,
            )
        else:
            inlier_mask_wrt_gt, reproj_error_wrt_gt = None, None
    else:
        R_error_deg, U_error_deg, inlier_mask_wrt_gt, reproj_error_wrt_gt = None, None, None, None

    return generate_two_view_report(
        inlier_ratio_wrt_estimate,
        verified_corr_idxs,
        R_error_deg=R_error_deg,
        U_error_deg=U_error_deg,
        v_corr_idxs_inlier_mask_gt=inlier_mask_wrt_gt,
        reproj_error_gt_model=reproj_error_wrt_gt,
    )


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


def compute_relative_pose_metrics(
    i2Ri1_computed: Optional[Rot3],
    i2Ui1_computed: Optional[Unit3],
    wTi1_expected: Optional[Pose3],
    wTi2_expected: Optional[Pose3],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute the metrics on relative camera pose.

    Args:
        i2Ri1_computed: Computed relative rotation.
        i2Ui1_computed: Computed relative translation direction.
        i2Ti1_expected: Expected relative pose.

    Returns:
        Rotation error, in degrees
        Unit translation error, in degrees
    """
    if wTi1_expected is not None and wTi2_expected is not None:
        i2Ti1_expected = wTi2_expected.between(wTi1_expected)
        R_error_deg = comp_utils.compute_relative_rotation_angle(i2Ri1_computed, i2Ti1_expected.rotation())
        U_error_deg = comp_utils.compute_relative_unit_translation_angle(
            i2Ui1_computed, Unit3(i2Ti1_expected.translation())
        )
    else:
        return (None, None)

    return (R_error_deg, U_error_deg)


def aggregate_frontend_metrics(
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    angular_err_threshold_deg: float,
    metric_group_name: str,
) -> GtsfmMetricsGroup:
    """Aggregate the front-end metrics to log summary statistics.

    We define "pose error" as the maximum of the angular errors in rotation and translation, per:
        SuperGlue, CVPR 2020: https://arxiv.org/pdf/1911.11763.pdf
        Learning to find good correspondences. CVPR 2018:
        OA-Net, ICCV 2019:
        NG-RANSAC, ICCV 2019:

    Args:
        two_view_report_dict: Report containing front-end metrics for each image pair.
        angular_err_threshold_deg: Threshold for classifying angular error metrics as success.
        metric_group_name: Name we will assign to the GtsfmMetricGroup returned by this fn.
    """
    num_image_pairs = len(two_view_reports_dict.keys())

    # all rotational errors in degrees
    rot3_angular_errors_list: List[float] = []
    trans_angular_errors_list: List[float] = []

    inlier_ratio_gt_model_all_pairs = []
    inlier_ratio_est_model_all_pairs = []
    num_inliers_gt_model_all_pairs = []
    num_inliers_est_model_all_pairs = []
    # populate the distributions
    for report in two_view_reports_dict.values():
        if report.R_error_deg is not None:
            rot3_angular_errors_list.append(report.R_error_deg)
        if report.U_error_deg is not None:
            trans_angular_errors_list.append(report.U_error_deg)

        inlier_ratio_gt_model_all_pairs.append(report.inlier_ratio_gt_model)
        inlier_ratio_est_model_all_pairs.append(report.inlier_ratio_est_model)
        num_inliers_gt_model_all_pairs.append(report.num_inliers_gt_model)
        num_inliers_est_model_all_pairs.append(report.num_inliers_est_model)

    rot3_angular_errors = np.array(rot3_angular_errors_list, dtype=float)
    trans_angular_errors = np.array(trans_angular_errors_list, dtype=float)
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


def run_two_view_estimator_as_futures(
    client: Client,
    two_view_estimator: TwoViewEstimator,
    keypoints_list: List[Keypoints],
    putative_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    camera_intrinsics: List[gtsfm_types.CALIBRATION_TYPE],
    relative_pose_priors: Dict[Tuple[int, int], PosePrior],
    gt_cameras: List[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_scene_mesh: Optional[Any],
) -> Dict[Tuple[int, int], TWO_VIEW_OUTPUT]:
    """Run two-view estimator for all image pairs."""

    def apply_two_view_estimator(
        two_view_estimator: TwoViewEstimator,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        putative_corr_idxs: np.ndarray,
        camera_intrinsics_i1: gtsfm_types.CALIBRATION_TYPE,
        camera_intrinsics_i2: gtsfm_types.CALIBRATION_TYPE,
        i2Ti1_prior: Optional[PosePrior],
        gt_camera_i1: Optional[gtsfm_types.CAMERA_TYPE],
        gt_camera_i2: Optional[gtsfm_types.CAMERA_TYPE],
        gt_scene_mesh: Optional[Any] = None,
    ) -> TWO_VIEW_OUTPUT:
        return two_view_estimator.run_2view(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            putative_corr_idxs=putative_corr_idxs,
            camera_intrinsics_i1=camera_intrinsics_i1,
            camera_intrinsics_i2=camera_intrinsics_i2,
            i2Ti1_prior=i2Ti1_prior,
            gt_camera_i1=gt_camera_i1,
            gt_camera_i2=gt_camera_i2,
            gt_scene_mesh=gt_scene_mesh,
        )

    two_view_estimator_future = client.scatter(two_view_estimator, broadcast=False)

    two_view_output_futures = {
        (i1, i2): client.submit(
            apply_two_view_estimator,
            two_view_estimator_future,
            keypoints_list[i1],
            keypoints_list[i2],
            putative_corr_idxs,
            camera_intrinsics[i1],
            camera_intrinsics[i2],
            relative_pose_priors.get((i1, i2)),
            gt_cameras[i1],
            gt_cameras[i2],
            gt_scene_mesh,
        )
        for (i1, i2), putative_corr_idxs in putative_corr_idxs_dict.items()
    }

    two_view_output_dict = client.gather(two_view_output_futures)

    return two_view_output_dict
