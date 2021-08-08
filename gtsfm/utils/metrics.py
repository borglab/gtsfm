"""Utilities to compute and save evaluation metrics.

Authors: Ayush Baid, Akshay Krishnan
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, EssentialMatrix, Point3, Pose3, Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.two_view_estimator import TwoViewEstimationReport

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# A StatsDict is a dict from string to optional floats or their lists.
StatsDict = Dict[str, Union[Optional[float], List[Optional[float]]]]

"""
data type for frontend metrics on a pair of images, containing:
1. rotation angular error
2. translation angular error
3. number of correct correspondences
4. inlier ratio
"""
FRONTEND_METRICS_FOR_PAIR = Tuple[Optional[float], Optional[float], int, float]

# number of digits (significant figures) to include in each entry of error metrics
PRINT_NUM_SIG_FIGS = 2

METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "result_metrics"
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "rtf_vis_tool" / "src" / "result_metrics"


logger = logger_utils.get_logger()


def count_correct_correspondences(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    intrinsics_i1: Cal3Bundler,
    intrinsics_i2: Cal3Bundler,
    i2Ti1: Pose3,
    epipolar_dist_threshold: float,
) -> int:
    """Checks the correspondences for epipolar distances and counts ones which are below the threshold.

    Args:
        keypoints_i1: keypoints in image i1.
        keypoints_i2: corr. keypoints in image i2.
        intrinsics_i1: intrinsics for i1.
        intrinsics_i2: intrinsics for i2.
        i2Ti1: relative pose
        epipolar_dist_threshold: max acceptable distance for a correct correspondence.

    Raises:
        ValueError: when the number of keypoints do not match.

    Returns:
        Number of correspondences which are correct.
    """
    # TODO: add unit test, with mocking.
    if len(keypoints_i1) != len(keypoints_i2):
        raise ValueError("Keypoints must have same counts")

    if len(keypoints_i1) == 0:
        return 0

    i2Ei1 = EssentialMatrix(i2Ti1.rotation(), Unit3(i2Ti1.translation()))
    i2Fi1 = verification_utils.essential_to_fundamental_matrix(i2Ei1, intrinsics_i1, intrinsics_i2)

    distance_squared = verification_utils.compute_epipolar_distances_sq_sampson(
        keypoints_i1.coordinates, keypoints_i2.coordinates, i2Fi1
    )
    return np.count_nonzero(distance_squared < epipolar_dist_threshold ** 2)


def compute_rotation_angle_metric(wRi_list: List[Optional[Rot3]], gt_wRi_list: List[Optional[Pose3]]) -> GtsfmMetric:
    """Computes statistics for the angle between estimated and GT rotations.

    Assumes that the estimated and GT rotations have been aligned and do not
    have a gauge freedom.

    Args:
        wRi_list: List of estimated camera rotations.
        gt_wRi_list: List of ground truth camera rotations.

    Returns:
        A statistics dict of the metrics errors in degrees.
    """
    errors = []
    for (wRi, gt_wRi) in zip(wRi_list, gt_wRi_list):
        if wRi is not None and gt_wRi is not None:
            errors.append(comp_utils.compute_relative_rotation_angle(wRi, gt_wRi))
    return GtsfmMetric("rotation_averaging_angle_deg", np.array(errors))


def compute_translation_distance_metric(
    wti_list: List[Optional[Point3]], gt_wti_list: List[Optional[Point3]]
) -> GtsfmMetric:
    """Computes statistics for the distance between estimated and GT translations.

    Assumes that the estimated and GT translations have been aligned and do not
    have a gauge freedom (including scale).

    Args:
        wti_list: List of estimated camera translations.
        gt_wti_list: List of ground truth camera translations.

    Returns:
        A statistics dict of the metrics errors in degrees.
    """
    errors = []
    for (wti, gt_wti) in zip(wti_list, gt_wti_list):
        if wti is not None and gt_wti is not None:
            errors.append(comp_utils.compute_points_distance_l2(wti, gt_wti))
    return GtsfmMetric("translation_averaging_distance", np.array(errors))


def compute_translation_angle_metric(
    i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]], wTi_list: List[Optional[Pose3]]
) -> GtsfmMetric:
    """Computes statistics for angle between translations and direction measurements.

    Args:
        i2Ui1_dict: List of translation direction measurements.
        wTi_list: List of estimated camera poses.

    Returns:
        A statistics dict of the metrics errors in degrees.
    """
    angles = []
    for (i1, i2) in i2Ui1_dict:
        i2Ui1 = i2Ui1_dict[(i1, i2)]
        angles.append(comp_utils.compute_translation_to_direction_angle(i2Ui1, wTi_list[i2], wTi_list[i1]))
    return GtsfmMetric("translation_to_direction_angle_deg", np.array(angles))


def compute_averaging_metrics(
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    gt_wTi_list: List[Optional[Pose3]],
) -> GtsfmMetricsGroup:
    """Computes statistics of multiple metrics for the averaging modules.

    Specifically, computes statistics of:
        - Rotation angle errors before BA,
        - Translation distances before BA,
        - Translation angle to direction measurements,

    Estimated poses and ground truth poses are first aligned before computing metrics.

    Args:
        i2Ui1_dict: Dict from (i1, i2) to unit translation measurement i2Ui1.
        wRi_list: List of estimated rotations.
        wti_list: List of estimated translations.
        gt_wTi_list: List of ground truth poses.

    Returns:
        Dict from metric name to a StatsDict.

    Raises:
        ValueError if lengths of wRi_list, wti_list and gt_wTi_list are not all same.
    """
    if len(wRi_list) != len(wti_list) or len(wRi_list) != len(gt_wTi_list):
        raise ValueError("Lengths of wRi_list, wti_list and gt_wTi_list should be the same.")

    wTi_list = []
    for (wRi, wti) in zip(wRi_list, wti_list):
        # if translation estimation failed in translation averaging, some wti_list values will be None
        if wRi is None or wti is None:
            wTi_list.append(None)
        else:
            wTi_list.append(Pose3(wRi, wti))

    # ground truth is the reference/target for alignment
    wTi_aligned_list = comp_utils.align_poses_sim3_ignore_missing(gt_wTi_list, wTi_list)

    wRi_aligned_list, wti_aligned_list = get_rotations_translations_from_poses(wTi_aligned_list)
    gt_wRi_list, gt_wti_list = get_rotations_translations_from_poses(gt_wTi_list)

    metrics = []
    metrics.append(compute_rotation_angle_metric(wRi_aligned_list, gt_wRi_list))
    metrics.append(compute_translation_distance_metric(wti_aligned_list, gt_wti_list))
    metrics.append(compute_translation_angle_metric(i2Ui1_dict, wTi_aligned_list))
    return GtsfmMetricsGroup("averaging_metrics", metrics)


def get_rotations_translations_from_poses(
    poses: List[Optional[Pose3]],
) -> Tuple[List[Optional[Rot3]], List[Optional[Point3]]]:
    """Decompose each 6-dof pose to a 3-dof rotation and 3-dof position"""
    rotations = []
    translations = []
    for pose in poses:
        if pose is None:
            rotations.append(None)
            translations.append(None)
            continue
        rotations.append(pose.rotation())
        translations.append(pose.translation())
    return rotations, translations


def log_sfm_summary() -> None:
    """Dump to stdout a summary of metrics about the SfM reconstruction process."""
    frontend_full_metrics_fpath = REPO_ROOT / "result_metrics" / "frontend_full.json"
    frontend_metrics = io_utils.read_json_file(frontend_full_metrics_fpath)

    rot_errs_deg = [
        pair_stats["rotation_angular_error"] for pair_stats in frontend_metrics if pair_stats["rotation_angular_error"]
    ]
    trans_errs_deg = [
        pair_stats["translation_angular_error"]
        for pair_stats in frontend_metrics
        if pair_stats["translation_angular_error"]
    ]

    logger.info("=============> Metrics report ==============>")
    logger.info("Front-end median_rot_err_deg: %.2f", np.median(rot_errs_deg))
    logger.info("Front-end max_rot_err_deg: %.2f", max(rot_errs_deg))

    logger.info("Front-end median_trans_err_deg: %.2f", np.median(trans_errs_deg))
    logger.info("Front-end max_trans_err_deg: %.2f", max(trans_errs_deg))

    averaging_metrics_fpath = REPO_ROOT / "result_metrics" / "multiview_optimizer_metrics.json"
    averaging_metrics = io_utils.read_json_file(averaging_metrics_fpath)

    logger.info("Averaging median_rot_err_deg: %.2f", averaging_metrics["rotation_averaging_angle_deg"]["median_error"])
    logger.info("Averaging max_rot_err_deg: %.2f", averaging_metrics["rotation_averaging_angle_deg"]["max_error"])

    logger.info(
        "Averaging median_trans_dist_err: %.2f", averaging_metrics["translation_averaging_distance"]["median_error"]
    )
    logger.info("Averaging max_trans_dist_err: %.2f", averaging_metrics["translation_averaging_distance"]["max_error"])


def persist_frontend_metrics_full(
    two_view_report_dict: Dict[Tuple[int, int], TwoViewEstimationReport], images: List[Image]
) -> None:
    """Persist the front-end metrics for every pair on disk.

    Args:
        two_view_report_dict: front-end metrics for pairs of images.
        images: list of all images for this scene, in order of image/frame index.
    """
    metrics_list = []

    for (i1, i2), report in two_view_report_dict.items():

        # Note: if GT is unknown, then R_error_deg, U_error_deg, and inlier_ratio_gt_model will be None
        metrics_list.append(
            {
                "i1": i1,
                "i2": i2,
                "i1_filename": images[i1].file_name,
                "i2_filename": images[i2].file_name,
                "rotation_angular_error": round(report.R_error_deg, PRINT_NUM_SIG_FIGS) if report.R_error_deg else None,
                "translation_angular_error": round(report.U_error_deg, PRINT_NUM_SIG_FIGS)
                if report.U_error_deg
                else None,
                "num_inliers_gt_model": report.num_inliers_gt_model if report.num_inliers_gt_model else None,
                "inlier_ratio_gt_model": round(report.inlier_ratio_gt_model, PRINT_NUM_SIG_FIGS)
                if report.inlier_ratio_gt_model
                else None,
                "inlier_ratio_est_model": round(report.inlier_ratio_est_model, PRINT_NUM_SIG_FIGS),
                "num_inliers_est_model": report.num_inliers_est_model,
            }
        )

    io_utils.save_json_file(os.path.join(METRICS_PATH, "frontend_full.json"), metrics_list)

    # Save duplicate copy of 'frontend_full.json' within React Folder.
    io_utils.save_json_file(os.path.join(REACT_METRICS_PATH, "frontend_full.json"), metrics_list)


def aggregate_frontend_metrics(
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport], angular_err_threshold_deg: float
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
    """
    num_image_pairs = len(two_view_reports_dict.keys())

    # all rotational errors in degrees
    rot3_angular_errors = [report.R_error_deg for report in two_view_reports_dict.values()]
    trans_angular_errors = [report.U_error_deg for report in two_view_reports_dict.values()]
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

    frontend_metrics = GtsfmMetricsGroup("frontend_summary", [
        GtsfmMetric("angular_err_threshold_deg", angular_err_threshold_deg), 
        GtsfmMetric("num_valid_image_pairs", int(num_valid_image_pairs)), 
        GtsfmMetric("rotation_success_count", int(success_count_rot3)), 
        GtsfmMetric("translation_success_count", int(success_count_unit3)), 
        GtsfmMetric("pose_success_count", int(success_count_pose)), 
        GtsfmMetric("num_all_inlier_correspondeces_wrt_gt_model", int(all_correct))
    ])
    return frontend_metrics


def save_metrics_as_json(metrics_groups: Delayed, output_dir: str) -> None:
    for metrics_group in metrics_groups:
        metrics_group.save_to_json(os.path.join(output_dir, metrics_group.name + ".json"))
