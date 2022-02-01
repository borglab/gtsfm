"""Estimator which operates on a pair of images to compute relative pose and verified indices.

Authors: Ayush Baid, John Lambert
"""
import logging
import timeit
from typing import Dict, Optional, Tuple, List

import dask
import gtsam
import numpy as np
from dask.delayed import Delayed
from gtsam import (
    Cal3Bundler,
    CameraSetCal3Bundler,
    Point2Vector,
    Pose3,
    PinholeCameraCal3Bundler,
    Rot3,
    SfmTrack,
    Unit3,
)

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport, ConfigurationType
from gtsfm.data_association.point3d_initializer import SVD_DLT_RANK_TOL
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.verifier.verifier_base import VerifierBase
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


class TwoViewEstimator:
    """Wrapper for running two-view relative pose estimation on image pairs in the dataset."""

    def __init__(
        self,
        matcher: MatcherBase,
        verifier: VerifierBase,
        inlier_support_processor: InlierSupportProcessor,
        bundle_adjust_2view: bool,
        eval_threshold_px: float,
        bundle_adjust_2view_maxiters: int = 100,
    ) -> None:
        """Initializes the two-view estimator from matcher and verifier.

        Args:
            matcher: matcher to use.
            verifier: verifier to use.
            inlier_support_processor: post-processor that uses information about RANSAC support to filter out pairs.
            bundle_adjust_2view: boolean flag indicating if bundle adjustment is to be run on the 2-view data.
            eval_threshold_px: distance threshold for marking a correspondence pair as inlier during evaluation
                (not during estimation).
            bundle_adjust_2view_maxiters (optional): max number of iterations for 2-view BA. Defaults to 100.
        """
        self._matcher = matcher
        self._verifier = verifier
        self.processor = inlier_support_processor
        self._bundle_adjust_2view = bundle_adjust_2view
        self._corr_metric_dist_threshold = eval_threshold_px
        self._ba_optimizer = BundleAdjustmentOptimizer(
            robust_measurement_noise=True, max_iterations=bundle_adjust_2view_maxiters
        )

    @classmethod
    def triangulate_two_view_correspondences(
        cls,
        camera_i1: PinholeCameraCal3Bundler,
        camera_i2: PinholeCameraCal3Bundler,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        corr_idxs: np.ndarray,
    ):
        """Triangulate 2-view correspondences to form 3d tracks.

        Args:
            camera_i1: camera for 1st view.
            camera_i2: camera for 2nd view.
            keypoints_i1: keypoints for 1st view.
            keypoints_i2: keypoints for 2nd view.
            corr_idxs: indices of corresponding keypoints.

        Returns:
            Triangulated 3D points.
        """
        camera_set = CameraSetCal3Bundler()
        camera_set.append(camera_i1)
        camera_set.append(camera_i2)

        tracks_3d: List[SfmTrack] = []
        for i in range(len(corr_idxs)):
            track_2d = Point2Vector()
            idx1, idx2 = corr_idxs[i, :]
            track_2d.append(keypoints_i1.coordinates[idx1])
            track_2d.append(keypoints_i2.coordinates[idx2])

            try:
                triangulated_pt = gtsam.triangulatePoint3(
                    camera_set, track_2d, rank_tol=SVD_DLT_RANK_TOL, optimize=False
                )
                track_3d = SfmTrack(triangulated_pt)
                track_3d.add_measurement(0, track_2d[0])
                track_3d.add_measurement(1, track_2d[1])
                tracks_3d.append(track_3d)
            except RuntimeError:
                pass

        return tracks_3d

    def bundle_adjust(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        verified_corr_idxs: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
        i2Ri1_initial: Optional[Rot3],
        i2Ui1_initial: Optional[Unit3],
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Refine the relative pose using bundle adjustment on the 2-view scene.

        Args:
            keypoints_i1: keypoints from image i1.
            keypoints_i2: keypoints from image i2.
            verified_corr_idxs: indices of verified correspondences between i1 and i2.
            camera_intrinsics_i1: intrinsics for i1.
            camera_intrinsics_i2: intrinsics for i2.
            i2Ri1_initial: the relative rotation to be used as initial rotation between cameras.
            i2Ui1_initial: the relative unit direction, to be used to initialize initial translation between cameras.
        Returns:
            Optimized relative rotation i2Ri1.
            Optimized unit translation i2Ui1.
            Optimized verified_corr_idxs.
        """
        if i2Ri1_initial is None or i2Ui1_initial is None:
            return None, None, verified_corr_idxs

        i2Ti1_initial = Pose3(i2Ri1_initial, i2Ui1_initial.point3())

        # Set the i1 camera pose as the global coordinate system.
        camera_i1 = PinholeCameraCal3Bundler(Pose3(), camera_intrinsics_i1)
        camera_i2 = PinholeCameraCal3Bundler(i2Ti1_initial.inverse(), camera_intrinsics_i2)

        # Perform data association to construct 2-view BA input.
        start_time = timeit.default_timer()
        triangulated_tracks: List[SfmTrack] = self.triangulate_two_view_correspondences(
            camera_i1=camera_i1,
            camera_i2=camera_i2,
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            corr_idxs=verified_corr_idxs,
        )
        logger.debug("Performed DA in %.6f seconds.", timeit.default_timer() - start_time)

        # Perform 2-view BA.
        start_time = timeit.default_timer()
        ba_input = GtsfmData(number_images=2)
        ba_input.add_camera(0, camera_i1)
        ba_input.add_camera(1, camera_i2)
        for track in triangulated_tracks:
            ba_input.add_track(track)
        ba_output, _ = self._ba_optimizer.run(ba_input, verbose=False)
        wTi1, wTi2 = ba_output.get_camera_poses()  # extract the camera poses
        if wTi1 is None or wTi2 is None:
            logger.warning("2-view BA failed")
            return i2Ri1_initial, i2Ui1_initial, verified_corr_idxs
        i2Ti1_optimized = wTi2.between(wTi1)
        logger.debug("Performed 2-view BA in %.6f seconds.", timeit.default_timer() - start_time)

        return i2Ti1_optimized.rotation(), Unit3(i2Ti1_optimized.translation()), verified_corr_idxs

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
    ) -> Tuple[Delayed, Delayed, Delayed, Dict[str, Delayed]]:
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
            Two-view reports at different stages (pre BA, post BA, and post inlier-support-processor), as a dictionary.
        """

        # graph for matching to obtain putative correspondences
        putative_corr_idxs = self._matcher.create_computation_graph(
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
            pre_ba_i2Ri1,
            pre_ba_i2Ui1,
            pre_ba_v_corr_idxs,
            inlier_ratio_wrt_estimate,
            configuration_type,
        ) = self._verifier.create_computation_graph(
            keypoints_i1_graph,
            keypoints_i2_graph,
            putative_corr_idxs,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
        )

        if self._bundle_adjust_2view:
            post_ba_i2Ri1, post_ba_i2Ui1, post_ba_v_corr_idxs = dask.delayed(self.bundle_adjust, nout=3)(
                keypoints_i1_graph,
                keypoints_i2_graph,
                pre_ba_v_corr_idxs,
                camera_intrinsics_i1_graph,
                camera_intrinsics_i2_graph,
                pre_ba_i2Ri1,
                pre_ba_i2Ui1,
            )
        else:
            post_ba_i2Ri1 = pre_ba_i2Ri1
            post_ba_i2Ui1 = pre_ba_i2Ui1
            post_ba_v_corr_idxs = pre_ba_v_corr_idxs

        # if we have the expected GT data, evaluate the computed relative pose
        if gt_wTi1_graph is not None and gt_wTi2_graph is not None:
            i2Ti1_expected_graph = gt_wTi2_graph.between(gt_wTi1_graph)
            pre_ba_R_error_deg, pre_ba_U_error_deg = dask.delayed(compute_relative_pose_metrics, nout=2)(
                pre_ba_i2Ri1, pre_ba_i2Ui1, i2Ti1_expected_graph
            )
            post_ba_R_error_deg, post_ba_U_error_deg = dask.delayed(compute_relative_pose_metrics, nout=2)(
                post_ba_i2Ri1, post_ba_i2Ui1, i2Ti1_expected_graph
            )
            pre_ba_inlier_mask_wrt_gt, pre_ba_reproj_error_wrt_gt = dask.delayed(
                metric_utils.compute_correspondence_metrics, nout=2
            )(
                keypoints_i1_graph,
                keypoints_i2_graph,
                pre_ba_v_corr_idxs,
                camera_intrinsics_i1_graph,
                camera_intrinsics_i2_graph,
                self._corr_metric_dist_threshold,
                gt_wTi1_graph,
                gt_wTi2_graph,
                gt_scene_mesh_graph,
            )
            post_ba_inlier_mask_wrt_gt, post_ba_reproj_error_wrt_gt = dask.delayed(
                metric_utils.compute_correspondence_metrics, nout=2
            )(
                keypoints_i1_graph,
                keypoints_i2_graph,
                post_ba_v_corr_idxs,
                camera_intrinsics_i1_graph,
                camera_intrinsics_i2_graph,
                self._corr_metric_dist_threshold,
                gt_wTi1_graph,
                gt_wTi2_graph,
                gt_scene_mesh_graph,
            )
        else:
            pre_ba_R_error_deg, pre_ba_U_error_deg = None, None
            post_ba_R_error_deg, post_ba_U_error_deg = None, None
            pre_ba_inlier_mask_wrt_gt, pre_ba_reproj_error_wrt_gt = None, None
            post_ba_inlier_mask_wrt_gt, post_ba_reproj_error_wrt_gt = None, None

        pre_ba_report = dask.delayed(generate_two_view_report)(
            configuration_type=configuration_type,
            inlier_ratio_est_model=inlier_ratio_wrt_estimate,
            v_corr_idxs=pre_ba_v_corr_idxs,
            R_error_deg=pre_ba_R_error_deg,
            U_error_deg=pre_ba_U_error_deg,
            v_corr_idxs_inlier_mask_gt=pre_ba_inlier_mask_wrt_gt,
            reproj_error_gt_model=pre_ba_reproj_error_wrt_gt,
        )

        post_ba_report = dask.delayed(generate_two_view_report)(
            configuration_type=configuration_type,
            inlier_ratio_est_model=inlier_ratio_wrt_estimate,  # TODO: dont store ratios so that we can update them
            v_corr_idxs=post_ba_v_corr_idxs,
            R_error_deg=post_ba_R_error_deg,
            U_error_deg=post_ba_U_error_deg,
            v_corr_idxs_inlier_mask_gt=post_ba_inlier_mask_wrt_gt,
            reproj_error_gt_model=post_ba_reproj_error_wrt_gt,
        )

        (
            post_isp_i2Ri1,
            post_isp_i2Ui1,
            post_isp_v_corr_idxs,
            post_isp_report,
        ) = self.processor.create_computation_graph(post_ba_i2Ri1, post_ba_i2Ui1, post_ba_v_corr_idxs, post_ba_report)

        two_view_reports = {
            PRE_BA_REPORT_TAG: pre_ba_report,
            POST_BA_REPORT_TAG: post_ba_report,
            POST_ISP_REPORT_TAG: post_isp_report,
        }

        return post_isp_i2Ri1, post_isp_i2Ui1, post_isp_v_corr_idxs, two_view_reports


def generate_two_view_report(
    configuration_type: ConfigurationType,
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
        configuration_type=configuration_type,
    )
    return two_view_report


def compute_relative_pose_metrics(
    i2Ri1_computed: Optional[Rot3], i2Ui1_computed: Optional[Unit3], i2Ti1_expected: Pose3
) -> Tuple[Optional[float], Optional[float]]:
    """Compute the metrics on relative camera pose.

    Args:
        i2Ri1_computed: computed relative rotation.
        i2Ui1_computed: computed relative translation direction.
        i2Ti1_expected: expected relative pose.

    Returns:
        Rotation error, in degrees.
        Unit translation error, in degrees. For a panoramic image pair configuration,
            the angular error is defined (NaN).
    """
    R_error_deg = comp_utils.compute_relative_rotation_angle(i2Ri1_computed, i2Ti1_expected.rotation())

    # Same check as in GTSAM: https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/TranslationRecovery.cpp#L52
    # and in COLMAP: https://github.com/colmap/colmap/blob/dev/src/estimators/two_view_geometry.cc#L221
    if np.allclose(i2Ui1_computed.point3(), np.zeros(3)):
        # panoramic case, there is no direction, so we cannot measure directional error (undefined!)
        U_error_deg = np.nan
    else:
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
    rot3_angular_errors: List[float] = []
    trans_angular_errors: List[float] = []

    inlier_ratio_gt_model_all_pairs = []
    inlier_ratio_est_model_all_pairs = []
    num_inliers_gt_model_all_pairs = []
    num_inliers_est_model_all_pairs = []

    from collections import defaultdict

    configuration_type_counts = defaultdict(int)
    # populate the distributions
    for report in two_view_reports_dict.values():
        if report is None:
            continue
        if report.R_error_deg is not None:
            rot3_angular_errors.append(report.R_error_deg)
        if report.U_error_deg is not None:
            trans_angular_errors.append(report.U_error_deg)

        inlier_ratio_gt_model_all_pairs.append(report.inlier_ratio_gt_model)
        inlier_ratio_est_model_all_pairs.append(report.inlier_ratio_est_model)
        num_inliers_gt_model_all_pairs.append(report.num_inliers_gt_model)
        num_inliers_est_model_all_pairs.append(report.num_inliers_est_model)

        configuration_type_counts[report.configuration_type] += 1

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

    metrics_list = [
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
    ]

    # for config_type in gric_verifier.ConfigurationType:
    logger.info("Configuration Type counts: " + str(configuration_type_counts))

    for config_type in ConfigurationType:
        metrics_list.append(GtsfmMetric(f"#{config_type} pairs", configuration_type_counts[config_type]))

    # TODO(akshay-krishnan): Move angular_err_threshold_deg and num_total_image_pairs to metadata.
    frontend_metrics = GtsfmMetricsGroup(metric_group_name, metrics_list)
    return frontend_metrics
