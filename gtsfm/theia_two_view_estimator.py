"""Estimator which operates on a pair of images to compute relative pose and verified indices.

Authors: Ayush Baid
"""
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import dask
import numpy as np
import pytheia as pt
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Pose3, Rot3, Unit3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport, TheiaTwoViewInfo
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

PRE_BA_REPORT_TAG = "PRE_BA_2VIEW_REPORT"
POST_BA_REPORT_TAG = "POST_BA_2VIEW_REPORT"
POST_ISP_REPORT_TAG = "POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT"
VIEWGRAPH_REPORT_TAG = "VIEWGRAPH_2VIEW_REPORT"

FAILURE_RESULT = (None, None, np.array([], dtype=np.uint64), 0.0)


class TheiaTwoViewEstimator:
    """Wrapper for running Theia's two-view relative pose estimation on image pairs in the dataset."""

    def __init__(
        self,
        eval_threshold_px: float,
    ) -> None:
        self._corr_metric_dist_threshold = eval_threshold_px
        self._min_num_inlier_matches = 100

    def __theia_correspondence_from_matches(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        putative_corr_idxs: np.ndarray,
    ) -> List[pt.matching.FeatureCorrespondence]:
        correspondences = []
        for idx1, idx2 in putative_corr_idxs:
            correspondences.append(
                pt.matching.FeatureCorrespondence(
                    pt.sfm.Feature(keypoints_i1.coordinates[idx1]), pt.sfm.Feature(keypoints_i2.coordinates[idx2])
                )
            )

        return correspondences

    def __theia_intrinsics(self, intrinsics: Optional[gtsfm_types.CALIBRATION_TYPE]) -> pt.sfm.CameraIntrinsicsPrior:
        assert isinstance(intrinsics, Cal3Bundler)

        prior = pt.sfm.CameraIntrinsicsPrior()
        prior.focal_length.value = [intrinsics.fx()]
        prior.aspect_ratio.value = [intrinsics.fy() / intrinsics.fx()]
        prior.principal_point.value = [intrinsics.px(), intrinsics.py()]
        prior.radial_distortion.value = [intrinsics.k1(), intrinsics.k2(), 0, 0]
        prior.tangential_distortion.value = [0, 0]
        prior.skew.value = [0]
        # TODO: unfix this
        # prior.image_width = int(760)
        # prior.image_height = int(1135)
        # 'PINHOLE_RADIAL_TANGENTIAL', 'DIVISION_UNDISTORTION', 'DOUBLE_SPHERE', 'FOV', 'EXTENDED_UNIFIED', 'FISHEYE
        prior.camera_intrinsics_model_type = "PINHOLE_RADIAL_TANGENTIAL"

        return prior

    def run_2view(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        putative_corr_idxs: np.ndarray,
        camera_intrinsics_i1: Optional[gtsfm_types.CALIBRATION_TYPE],
        camera_intrinsics_i2: Optional[gtsfm_types.CALIBRATION_TYPE],
        i2Ti1_prior: Optional[PosePrior],
        gt_wTi1: Optional[Pose3],
        gt_wTi2: Optional[Pose3],
        gt_scene_mesh: Optional[Any] = None,
    ) -> Tuple[
        Optional[Rot3],
        Optional[Unit3],
        np.ndarray,
        TwoViewEstimationReport,
        TwoViewEstimationReport,
        TwoViewEstimationReport,
    ]:
        """Estimate relative pose between two views, using verification."""
        # verification on putative correspondences to obtain relative pose and verified correspondences
        correspondences = self.__theia_correspondence_from_matches(keypoints_i1, keypoints_i2, putative_corr_idxs)

        prior1 = self.__theia_intrinsics(camera_intrinsics_i1)
        prior2 = self.__theia_intrinsics(camera_intrinsics_i2)

        options = pt.sfm.EstimateTwoViewInfoOptions()
        options.max_sampson_error_pixels = 1.0
        options.max_ransac_iterations = 250
        options.ransac_type = pt.sfm.RansacType(0)
        theia_two_view_result = pt.sfm.EstimateTwoViewInfo(
            options, prior1, prior2, correspondences
        )  # type: Tuple[bool, pt.sfm.TwoViewInfo, List[int]]
        success = theia_two_view_result[0]
        two_view_info = theia_two_view_result[1]
        inlier_indices = theia_two_view_result[2]

        i1Ri2_angleaxis = two_view_info.rotation_2
        i1Ri2_rot_angle = np.linalg.norm(i1Ri2_angleaxis)
        i1Ri2 = Rot3.AxisAngle(i1Ri2_angleaxis, i1Ri2_rot_angle)
        i1ti2 = two_view_info.position_2

        if not success or len(inlier_indices) < self._min_num_inlier_matches:
            success = False
            (pre_ba_i2Ri1, pre_ba_i2Ui1, verified_corr_idxs, inlier_ratio_wrt_estimate) = (
                None,
                None,
                np.array([], dtype=np.uint64),
                0.0,
            )
        else:
            i1Ti2 = Pose3(i1Ri2.inverse(), i1ti2)
            i2Ti1 = i1Ti2.inverse()
            # pre_ba_i2Ri1 = i2Ti1.rotation()
            pre_ba_i2Ui1 = Unit3(i2Ti1.translation())
            pre_ba_i2Ri1 = i1Ri2
            # pre_ba_i2Ui1 = Unit3(i1ti2)
            verified_corr_idxs = putative_corr_idxs[np.array(inlier_indices, dtype=np.uint32)]
            inlier_ratio_wrt_estimate = len(verified_corr_idxs) / len(putative_corr_idxs)

        # if we have the expected GT data, evaluate the computed relative pose
        pre_ba_R_error_deg, pre_ba_U_error_deg = compute_relative_pose_metrics(
            pre_ba_i2Ri1, pre_ba_i2Ui1, gt_wTi1, gt_wTi2
        )
        pre_ba_inlier_mask_wrt_gt, pre_ba_reproj_error_wrt_gt = metric_utils.compute_correspondence_metrics(
            keypoints_i1,
            keypoints_i2,
            verified_corr_idxs,
            camera_intrinsics_i1,
            camera_intrinsics_i2,
            self._corr_metric_dist_threshold,
            gt_wTi1,
            gt_wTi2,
            gt_scene_mesh,
        )
        pre_ba_report = generate_two_view_report(
            inlier_ratio_wrt_estimate,
            verified_corr_idxs,
            R_error_deg=pre_ba_R_error_deg,
            U_error_deg=pre_ba_U_error_deg,
            v_corr_idxs_inlier_mask_gt=pre_ba_inlier_mask_wrt_gt,
            reproj_error_gt_model=pre_ba_reproj_error_wrt_gt,
        )
        if success:
            pre_ba_report.theia_twoview_info = TheiaTwoViewInfo(
                focal_length_1=two_view_info.focal_length_1,
                focal_length_2=two_view_info.focal_length_2,
                position_2=two_view_info.position_2,
                rotation_2=two_view_info.rotation_2,
                num_verified_matches=two_view_info.num_verified_matches,
                num_homography_inliers=two_view_info.num_homography_inliers,
                visibility_score=two_view_info.visibility_score,
            )

        return (
            pre_ba_i2Ri1,
            pre_ba_i2Ui1,
            verified_corr_idxs,
            pre_ba_report,
            deepcopy(pre_ba_report),
            deepcopy(pre_ba_report),
        )

    def create_computation_graph(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        putative_corr_idxs: np.ndarray,
        camera_intrinsics_i1: Optional[gtsfm_types.CALIBRATION_TYPE],
        camera_intrinsics_i2: Optional[gtsfm_types.CALIBRATION_TYPE],
        i2Ti1_prior: Optional[PosePrior] = None,
        gt_wTi1: Optional[Pose3] = None,
        gt_wTi2: Optional[Pose3] = None,
        gt_scene_mesh_graph: Optional[Delayed] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, Dict[str, Delayed]]:
        """Create delayed tasks for two view geometry estimation, using verification.

        Args:
            keypoints_i1: keypoints for image i1.
            keypoints_i2: keypoints for image i2.
            putative_corr_idxs: putative correspondences between i1 and i2, as a Kx2 array.
            camera_intrinsics_i1: intrinsics for camera i1.
            camera_intrinsics_i2: intrinsics for camera i2.
            i2Ti1_prior: the prior on relative pose i2Ti1.
            i2Ti1_expected_graph (optional): ground truth relative pose, used for evaluation if available.

        Returns:
            Computed relative rotation wrapped as Delayed.
            Computed relative translation direction wrapped as Delayed.
            Indices of verified correspondences wrapped as Delayed.
            Two-view reports at different stages (pre BA, post BA, and post inlier-support-processor), as a dictionary.
        """
        (
            post_isp_i2Ri1,
            post_isp_i2Ui1,
            post_isp_v_corr_idxs,
            pre_ba_report,
            post_ba_report,
            post_isp_report,
        ) = dask.delayed(self.run_2view, nout=6)(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            putative_corr_idxs=putative_corr_idxs,
            camera_intrinsics_i1=camera_intrinsics_i1,
            camera_intrinsics_i2=camera_intrinsics_i2,
            i2Ti1_prior=i2Ti1_prior,
            gt_wTi1=gt_wTi1,
            gt_wTi2=gt_wTi2,
            gt_scene_mesh=gt_scene_mesh_graph,
        )
        # Return the reports as a dict of Delayed objects, instead of a single Delayed object.
        # This makes it countable and indexable.
        two_view_reports = {
            PRE_BA_REPORT_TAG: pre_ba_report,
            POST_BA_REPORT_TAG: post_ba_report,
            POST_ISP_REPORT_TAG: post_isp_report,
        }
        return post_isp_i2Ri1, post_isp_i2Ui1, post_isp_v_corr_idxs, two_view_reports


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
        i2Ri1_computed: computed relative rotation.
        i2Ui1_computed: computed relative translation direction.
        i2Ti1_expected: expected relative pose.

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
