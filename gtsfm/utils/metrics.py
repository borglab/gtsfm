import json
import math
import numpy as np
import os
import statistics
from typing import Delayed, List, Optional, Any

from gtsam import (Point3, Rot3, Pose3)

import gtsfm.utils.geometry_comparisons
from gtsfm.common.sfm_result import SfmResult


def get_metrics_from_errors(errors: List) -> Dict[str, Any]:
    metrics = {}
    valid_errors = [error for error in errors if error is not None]
    metrics["min_error"] = min(valid_errors)
    metrics["max_error"] = max(valid_errors)
    metrics["median_error"] = statistics.median(valid_errors)
    metrics["per_camera_errors"] = errors
    return metrics


def compute_rotation_angle_metrics(
        wRi_list: List[Optional[Rot3]],
        gt_wRi_list: List[Optional[Pose3]]) -> Dict[str, Any]:
    errors = []
    for (wRi, gt_wRi) in zip(wRi_list, gt_wRi_list):
        angle = geometry_comparisons.compute_relative_rotation_angle(
            wRi, gt_wRi) * 180 / math.pi
        errors.append(error)
    return get_metrics_from_errors(errors)


def compute_translation_distance_metrics(
        wti_list: List[Optional[Point3]],
        gt_wti_list: List[Optional[Point3]]) -> Dict[str, Any]:
    errors = []
    for (wti, gt_wti) in zip(wti_list, gt_wti_list):
        if wti is None or gt_wti is None:
            errors.append(None)
        else:
            errors.append(np.linalg.norm(wti - gt_wti))
    return get_metrics_from_errors(errors)


def compute_averaging_metrics(
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    gt_wTi_list: List[Optional[Pose3]],
) -> Dict[str, Any]:
    if len(wRi_list) != len(wti_list) or len(wRi_list) != len(gt_poses_list):
        raise AttributeError(
            "Lengths of wRi_list, wti_list and gt_wTi_list should be the same.")

    wTi_list = []
    for (wRi, wti) in zip(wRi_list, wti_list):
        wTi_list.append(Pose3(wRi, wti))
    wTi_aligned_list = geometry_comparisons.align_poses(
        wTi_aligned_list, gt_poses)

    def get_rotations_translations_from_poses(poses):
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

    wRi_aligned_list, wti_aligned_list = get_rotations_translations_from_poses(
        wTi_aligned_list)
    gt_wRi_list, gt_wti_list = get_rotations_translations_from_poses(
        gt_wTi_list)

    metrics = {}
    metrics['rotation_averaging_angle'] = compute_rotation_angle_metrics(
        wRi_aligned_list, gt_wRi_list)
    metrics[
        'translation_averaging_distance'] = compute_translation_distance_metrics(
            wTi_aligned_list, gt_wti_list)
    return metrics


def save_averaging_metrics(
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    gt_wTi_list: List[Optional[Pose3]],
    output_dir: str,
) -> None:
    metrics = compute_averaging_metrics(wRi_list, wti_list, gt_wTi_list)
    json_file_path = os.path.join(output_dir,
                                  'multiview_optimizer_metrics.json')

    with open(json_file_path) as json_file:
        json.dump(metrics, json_file)
