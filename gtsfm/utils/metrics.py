"""Utilities to compute and save evaluation metrics.

Authors: Ayush Baid, Akshay Krishnan
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Point3, Pose3, Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# A StatsDict is a dict from string to optional floats or their lists.
StatsDict = Dict[str, Union[Optional[float], List[Optional[float]]]]

# number of digits (significant figures) to include in each entry of error metrics
PRINT_NUM_SIG_FIGS = 2


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


def compute_errors_statistics(errors: List[Optional[float]]) -> StatsDict:
    """Computes statistics (min, max, median) on the given list of errors

    Args:
        errors: List of errors for a metric.

    Returns:
        A dict with keys min_error, max_error, median_error,
        and errors_list mapping to the respective stats.
    """
    metrics = {}
    valid_errors = [error for error in errors if error is not None]
    metrics["median_error"] = np.round(np.median(valid_errors), PRINT_NUM_SIG_FIGS)
    metrics["min_error"] = np.round(np.min(valid_errors), PRINT_NUM_SIG_FIGS)
    metrics["max_error"] = np.round(np.max(valid_errors), PRINT_NUM_SIG_FIGS)
    metrics["errors_list"] = [np.round(error, PRINT_NUM_SIG_FIGS) if error is not None else None for error in errors]
    return metrics


def compute_rotation_angle_metrics(wRi_list: List[Optional[Rot3]], gt_wRi_list: List[Optional[Pose3]]) -> StatsDict:
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
        errors.append(comp_utils.compute_relative_rotation_angle(wRi, gt_wRi))
    return compute_errors_statistics(errors)


def compute_translation_distance_metrics(
    wti_list: List[Optional[Point3]], gt_wti_list: List[Optional[Point3]]
) -> StatsDict:
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
        errors.append(comp_utils.compute_points_distance_l2(wti, gt_wti))
    return compute_errors_statistics(errors)


def compute_translation_angle_metrics(
    i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]], wTi_list: List[Optional[Pose3]]
) -> StatsDict:
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
    return compute_errors_statistics(angles)


def compute_averaging_metrics(
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    gt_wTi_list: List[Optional[Pose3]],
) -> Dict[str, StatsDict]:
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

    metrics = {}
    metrics["rotation_averaging_angle_deg"] = compute_rotation_angle_metrics(wRi_aligned_list, gt_wRi_list)
    metrics["translation_averaging_distance"] = compute_translation_distance_metrics(wti_aligned_list, gt_wti_list)
    metrics["translation_to_direction_angle_deg"] = compute_translation_angle_metrics(i2Ui1_dict, wTi_aligned_list)
    return metrics


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


def compute_pose_errors(gt_wTi_list: List[Pose3], wTi_list: List[Pose3]) -> Dict[str, StatsDict]:
    """Compare orientation and location errors for each estimated poses, vs. ground truth.

    Note: Poses must be aligned, before calling this function
    """
    wRi_list, wti_list = get_rotations_translations_from_poses(wTi_list)
    gt_wRi_list, gt_wti_list = get_rotations_translations_from_poses(gt_wTi_list)

    metrics = {}
    metrics["rotation_angle_deg_errors"] = compute_rotation_angle_metrics(wRi_list, gt_wRi_list)
    metrics["translation_distance_errors"] = compute_translation_distance_metrics(wti_list, gt_wti_list)
    return metrics


def print_metrics() -> None:
    """Dump to stdout a summary of metrics about the SfM reconstruction process."""
    frontend_full_metrics_fpath = REPO_ROOT / "result_metrics" / "frontend_full.json"
    frontend_metrics = io_utils.read_json_file(frontend_full_metrics_fpath)

    rot_errs_deg = [pair_stats["rotation_angular_error"] for pair_stats in frontend_metrics if pair_stats["rotation_angular_error"]]
    trans_errs_deg = [pair_stats["translation_angular_error"] for pair_stats in frontend_metrics if pair_stats["translation_angular_error"]]

    logger.info("=============> Metrics report ==============>")
    logger.info("Front-end median_rot_err_deg: %.2f", np.median(rot_errs_deg))
    logger.info("Front-end max_rot_err_deg: %.2f", max(rot_errs_deg))

    logger.info("Front-end median_trans_err_deg: %.2f", np.median(trans_errs_deg))
    logger.info("Front-end max_trans_err_deg: %.2f", max(trans_errs_deg))

    averaging_metrics_fpath = REPO_ROOT / "result_metrics" / "multiview_optimizer_metrics.json"
    averaging_metrics = io_utils.read_json_file(averaging_metrics_fpath)

    logger.info("Averaging median_rot_err_deg: %.2f", averaging_metrics["rotation_averaging_angle_deg"]["median_error"])
    logger.info("Averaging max_rot_err_deg: %.2f", averaging_metrics["rotation_averaging_angle_deg"]["max_error"])

    logger.info("Averaging median_trans_dist_err: %.2f", averaging_metrics["translation_averaging_distance"]["median_error"])
    logger.info("Averaging max_trans_dist_err: %.2f", averaging_metrics["translation_averaging_distance"]["max_error"])

