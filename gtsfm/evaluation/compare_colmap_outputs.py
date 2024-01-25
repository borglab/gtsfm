"""Script to compare two reconstructions in Colmap's output format.

Authors: Ayush Baid
"""

import argparse
import os
from typing import Dict, List
from pathlib import Path

from gtsam import Pose3

import gtsfm.runner.gtsfm_runner_base as runner_base
import gtsfm.utils.geometry_comparisons as geometry_comparisons
import gtsfm.utils.io as io_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.evaluation.metrics import GtsfmMetricsGroup


def load_poses(colmap_dirpath: str) -> Dict[str, Pose3]:
    wTi_list, image_fnames = io_utils.read_images_txt(fpath=os.path.join(colmap_dirpath, "images.txt"))

    poses: Dict[str, Pose3] = {}
    for wTi, fname in zip(wTi_list, image_fnames):
        if wTi is None or fname is None:
            continue

        poses[fname] = wTi

    return poses


def compare_poses(baseline_dirpath: str, current_dirpath: str, output_path: str):
    baseline_wTi_dict = load_poses(baseline_dirpath)
    current_wTi_dict = load_poses(current_dirpath)

    common_fnames = baseline_wTi_dict.keys() & current_wTi_dict.keys()

    print(
        f"Baseline: {len(baseline_wTi_dict)}, current: {len(current_wTi_dict)} , common: {len(common_fnames)} entries"
    )

    baseline_wTi_list: List[Pose3] = []
    current_wTi_list: List[Pose3] = []
    for fname, wTi in baseline_wTi_dict.items():
        baseline_wTi_list.append(wTi)
        current_wTi_list.append(current_wTi_dict.get(fname))

    aligned_curr_wTi_list, _ = geometry_comparisons.align_poses_sim3_ignore_missing(baseline_wTi_list, current_wTi_list)

    i2Ui1_dict_gt = metric_utils.get_twoview_translation_directions(baseline_wTi_list)

    wRi_aligned_list, wti_aligned_list = metric_utils.get_rotations_translations_from_poses(aligned_curr_wTi_list)
    baseline_wRi_list, baseline_wti_list = metric_utils.get_rotations_translations_from_poses(baseline_wTi_list)

    metrics = []
    metrics.append(metric_utils.compute_rotation_angle_metric(wRi_aligned_list, baseline_wRi_list))
    metrics.append(metric_utils.compute_translation_distance_metric(wti_aligned_list, baseline_wti_list))
    metrics.append(metric_utils.compute_relative_translation_angle_metric(i2Ui1_dict_gt, aligned_curr_wTi_list))
    metrics.append(metric_utils.compute_translation_angle_metric(baseline_wTi_list, aligned_curr_wTi_list))

    rotation_angular_errors = metrics[0]._data
    translation_angular_errors = metrics[3]._data
    metrics.extend(
        metric_utils.compute_pose_auc_metric(rotation_angular_errors, translation_angular_errors, save_dir=output_path)
    )

    ba_pose_metrics = GtsfmMetricsGroup(name="ba_pose_error_metrics", metrics=metrics)

    runner_base.save_metrics_reports([ba_pose_metrics], metrics_path=output_path)


if __name__ == "__main__":
    """
    Compare two reconstructions (in Colmap's output format). Right now, we just compare the poses.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to directory containing benchmark artifacts for the baseline",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to directory containing benchmark artifacts for the current",
    )
    parser.add_argument("--output", required=True, help="Output for the json file for pose metrics")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    compare_poses(baseline_dirpath=args.baseline, current_dirpath=args.current, output_path=Path(args.output))
