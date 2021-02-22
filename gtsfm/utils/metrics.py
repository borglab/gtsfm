"""Utilities to compute and save evaluation metrics.

Authors: Ayush Baid, Akshay Krishnan
"""
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Point3, Pose3, Rot3, Unit3

import gtsfm.utils.features as feature_utils
import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints

# A StatsDict is a dict from string to optional floats or their lists.
StatsDict = Dict[str, Union[Optional[float], List[Optional[float]]]]


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

    normalized_coords_i1 = feature_utils.normalize_coordinates(keypoints_i1.coordinates, intrinsics_i1)
    normalized_coords_i2 = feature_utils.normalize_coordinates(keypoints_i2.coordinates, intrinsics_i2)
    i2Ei1 = EssentialMatrix(i2Ti1.rotation(), Unit3(i2Ti1.translation()))

    epipolar_distances = verification_utils.compute_epipolar_distances(
        normalized_coords_i1, normalized_coords_i2, i2Ei1
    )
    return np.count_nonzero(epipolar_distances < epipolar_dist_threshold)


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
    metrics["median_error"] = np.median(valid_errors)
    metrics["min_error"] = np.min(valid_errors)
    metrics["max_error"] = np.max(valid_errors)
    metrics["errors_list"] = errors
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
        wTi_list.append(Pose3(wRi, wti))
    wTi_aligned_list = comp_utils.align_poses(wTi_list, gt_wTi_list)

    def get_rotations_translations_from_poses(
        poses: List[Optional[Pose3]],
    ) -> Tuple[List[Optional[Rot3]], List[Optional[Point3]]]:
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

    wRi_aligned_list, wti_aligned_list = get_rotations_translations_from_poses(wTi_aligned_list)
    gt_wRi_list, gt_wti_list = get_rotations_translations_from_poses(gt_wTi_list)

    metrics = {}
    metrics["rotation_averaging_angle_deg"] = compute_rotation_angle_metrics(wRi_aligned_list, gt_wRi_list)
    metrics["translation_averaging_distance"] = compute_translation_distance_metrics(wti_aligned_list, gt_wti_list)
    metrics["translation_to_direction_angle_deg"] = compute_translation_angle_metrics(i2Ui1_dict, wTi_aligned_list)
    return metrics


def save_averaging_metrics(
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    gt_wTi_list: List[Optional[Pose3]],
    output_dir: str,
) -> None:
    """Computes the statistics of multiple metrics and saves them to json.

    Metrics are written to multiview_optimizer_metrics.json.

    Args:
        i2Ui1_dict: Dict from (i1, i2) to unit translation measurement i2Ui1.
        wRi_list: List of estimated rotations.
        wti_list: List of estimated translations.
        gt_wTi_list: List of ground truth poses.
        output_dir: Path to the directory where metrics must be saved.
    """
    metrics = compute_averaging_metrics(i2Ui1_dict, wRi_list, wti_list, gt_wTi_list)
    os.makedirs(output_dir, exist_ok=True)
    json_file_path = os.path.join(output_dir, "multiview_optimizer_metrics.json")

    with open(json_file_path, "w") as json_file:
        json.dump(metrics, json_file, indent=4)
