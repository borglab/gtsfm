"""Estimator which operates on a pair of images to compute relative pose and verified indices.

Authors: Ayush Baid, John Lambert
"""
import dataclasses
import logging
import timeit
from typing import Any, Dict, List, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import PinholeCameraCal3Bundler, Pose3, Rot3, SfmTrack, Unit3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.bundle.two_view_ba import TwoViewBundleAdjustment
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationOptions,
    TriangulationSamplingMode,
)
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.verifier.verifier_base import VerifierBase

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

    def bundle_adjust(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        verified_corr_idxs: np.ndarray,
        camera_intrinsics_i1: gtsfm_types.CALIBRATION_TYPE,
        camera_intrinsics_i2: gtsfm_types.CALIBRATION_TYPE,
        i2Ri1_initial: Optional[Rot3],
        i2Ui1_initial: Optional[Unit3],
        i2Ti1_prior: Optional[PosePrior],
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Refine the relative pose using bundle adjustment on the 2-view scene.

        Args:
            keypoints_i1: Keypoints from image i1.
            keypoints_i2: Keypoints from image i2.
            verified_corr_idxs: Indices of verified correspondences between i1 and i2.
            camera_intrinsics_i1: Intrinsics for i1.
            camera_intrinsics_i2: Intrinsics for i2.
            i2Ri1_initial: The relative rotation to be used as initial rotation between cameras.
            i2Ui1_initial: The relative unit direction, to be used to initialize initial translation between cameras.
            i2Ti1_prior: Prior on the relative pose for cameras (i1, i2).
        Returns:
            Optimized relative rotation i2Ri1.
            Optimized unit translation i2Ui1.
            Optimized verified_corr_idxs.
        """
        # Choose initial pose estimate for triangulation and BA (prior gets priority).
        i2Ti1_initial = i2Ti1_prior.value if i2Ti1_prior is not None else None
        if i2Ti1_initial is None and i2Ri1_initial is not None and i2Ui1_initial is not None:
            i2Ti1_initial = Pose3(i2Ri1_initial, i2Ui1_initial.point3())
        if i2Ti1_initial is None:
            return None, None, verified_corr_idxs

        # Set the i1 camera pose as the global coordinate system.
        camera_class = gtsfm_types.get_camera_class_for_calibration(camera_intrinsics_i1)
        cameras = {
            0: camera_class(Pose3(), camera_intrinsics_i1),
            1: camera_class(i2Ti1_initial.inverse(), camera_intrinsics_i2),
        }

        # Triangulate!
        point3d_initializer = Point3dInitializer(cameras, self._triangulation_options)
        triangulated_indices: List[int] = []
        triangulated_tracks: List[SfmTrack] = []
        start_time = timeit.default_timer()
        for j, (idx1, idx2) in enumerate(verified_corr_idxs):
            track2d = SfmTrack2d(
                [SfmMeasurement(0, keypoints_i1.coordinates[idx1]), SfmMeasurement(1, keypoints_i2.coordinates[idx2])]
            )
            track, _, _ = point3d_initializer.triangulate(track2d)
            if track is not None:
                triangulated_tracks.append(track)
                triangulated_indices.append(j)
        logger.debug("Performed DA in %.6f seconds.", timeit.default_timer() - start_time)
        logger.debug("Triangulated %d correspondences out of %d.", len(triangulated_tracks), len(verified_corr_idxs))

        if len(triangulated_tracks) == 0:
            return i2Ti1_initial.rotation(), Unit3(i2Ti1_initial.translation()), np.array([], dtype=np.uint32)

        # Build BA inputs.
        start_time = timeit.default_timer()
        ba_input = GtsfmData(number_images=2, cameras=cameras, tracks=triangulated_tracks)
        relative_pose_prior_for_ba = {(0, 1): i2Ti1_prior} if i2Ti1_prior is not None else {}

        # Optimize!
        _, ba_output, valid_mask = self._ba_optimizer.run_ba(
            ba_input, absolute_pose_priors=[], relative_pose_priors=relative_pose_prior_for_ba, verbose=False
        )

        # Unpack results.
        valid_corr_idxs = verified_corr_idxs[triangulated_indices][valid_mask]
        wTi1, wTi2 = ba_output.get_camera_poses()  # extract the camera poses
        if wTi1 is None or wTi2 is None:
            logger.warning("2-view BA failed...")
            return i2Ri1_initial, i2Ui1_initial, valid_corr_idxs
        i2Ti1_optimized = wTi2.between(wTi1)
        logger.debug("Performed 2-view BA in %.6f seconds.", timeit.default_timer() - start_time)

        return i2Ti1_optimized.rotation(), Unit3(i2Ti1_optimized.translation()), valid_corr_idxs

    def __get_2view_report_from_results(
        self,
        i2Ri1_computed: Optional[Rot3],
        i2Ui1_computed: Optional[Unit3],
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        verified_corr_idxs: np.ndarray,
        inlier_ratio_wrt_estimate: float,
        gt_camera_i1: Optional[gtsfm_types.CAMERA_TYPE],
        gt_camera_i2: Optional[gtsfm_types.CAMERA_TYPE],
        gt_scene_mesh: Optional[Any],
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
            if isinstance(gt_camera_i1, PinholeCameraCal3Bundler) and isinstance(
                gt_camera_i2, PinholeCameraCal3Bundler
            ):
                inlier_mask_wrt_gt, reproj_error_wrt_gt = metric_utils.compute_correspondence_metrics(
                    keypoints_i1=keypoints_i1,
                    keypoints_i2=keypoints_i2,
                    corr_idxs_i1i2=verified_corr_idxs,
                    dist_threshold=self._corr_metric_dist_threshold,
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
        (pre_ba_i2Ri1, pre_ba_i2Ui1, pre_ba_v_corr_idxs, pre_ba_inlier_ratio_wrt_estimate) = self._verifier.verify(
            keypoints_i1,
            keypoints_i2,
            putative_corr_idxs,
            camera_intrinsics_i1,
            camera_intrinsics_i2,
        )

        pre_ba_report = self.__get_2view_report_from_results(
            i2Ri1_computed=pre_ba_i2Ri1,
            i2Ui1_computed=pre_ba_i2Ui1,
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            verified_corr_idxs=pre_ba_v_corr_idxs,
            inlier_ratio_wrt_estimate=pre_ba_inlier_ratio_wrt_estimate,
            gt_camera_i1=gt_camera_i1,
            gt_camera_i2=gt_camera_i2,
            gt_scene_mesh=gt_scene_mesh,
        )

        # Optionally, do two-view bundle adjustment
        if self._bundle_adjust_2view:
            post_ba_i2Ri1, post_ba_i2Ui1, post_ba_v_corr_idxs = self.bundle_adjust(
                keypoints_i1,
                keypoints_i2,
                pre_ba_v_corr_idxs,
                camera_intrinsics_i1,
                camera_intrinsics_i2,
                pre_ba_i2Ri1,
                pre_ba_i2Ui1,
                i2Ti1_prior,
            )
            post_ba_inlier_ratio_wrt_estimate = float(len(post_ba_v_corr_idxs)) / len(putative_corr_idxs)

            # TODO: Remove this hack once we can handle the lower post_ba_inlier_ratio_wrt_estimate downstream.
            post_ba_inlier_ratio_wrt_estimate = pre_ba_inlier_ratio_wrt_estimate

            post_ba_report = self.__get_2view_report_from_results(
                i2Ri1_computed=post_ba_i2Ri1,
                i2Ui1_computed=post_ba_i2Ui1,
                keypoints_i1=keypoints_i1,
                keypoints_i2=keypoints_i2,
                verified_corr_idxs=post_ba_v_corr_idxs,
                inlier_ratio_wrt_estimate=post_ba_inlier_ratio_wrt_estimate,
                gt_camera_i1=gt_camera_i1,
                gt_camera_i2=gt_camera_i2,
                gt_scene_mesh=gt_scene_mesh,
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

        return post_isp_i2Ri1, post_isp_i2Ui1, post_isp_v_corr_idxs, pre_ba_report, post_ba_report, post_isp_report

    def create_computation_graph(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        putative_corr_idxs: np.ndarray,
        camera_intrinsics_i1: Optional[gtsfm_types.CALIBRATION_TYPE],
        camera_intrinsics_i2: Optional[gtsfm_types.CALIBRATION_TYPE],
        i2Ti1_prior: Optional[PosePrior] = None,
        gt_camera_i1: Optional[gtsfm_types.CAMERA_TYPE] = None,
        gt_camera_i2: Optional[gtsfm_types.CAMERA_TYPE] = None,
        gt_scene_mesh_graph: Optional[Delayed] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, Dict[str, Delayed]]:
        """Create delayed tasks for two view geometry estimation, using verification.

        Args:
            keypoints_i1: Keypoints for image i1.
            keypoints_i2: Keypoints for image i2.
            putative_corr_idxs: Putative correspondences between i1 and i2, as a Kx2 array.
            camera_intrinsics_i1: Intrinsics for camera i1.
            camera_intrinsics_i2: Intrinsics for camera i2.
            i2Ti1_prior: The prior on relative pose i2Ti1.
            i2Ti1_expected_graph (optional): Ground truth relative pose, used for evaluation if available.

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
            gt_camera_i1=gt_camera_i1,
            gt_camera_i2=gt_camera_i2,
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
) -> None:
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
