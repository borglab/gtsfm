"""Script to compare two reconstructions in Colmap's output format.

Authors: Ayush Baid
"""

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pycolmap
from gtsam import Pose3, Similarity3
from scipy.spatial.transform import Rotation

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.cluster_optimizer import save_metrics_reports
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.utils import align, transform

logger = logger_utils.get_logger()


def load_poses(colmap_dirpath: str) -> Dict[str, Pose3]:
    """Returns mapping from image filename to associated camera pose."""
    wTi_list, img_fnames, _, _, _, _ = io_utils.read_scene_data_from_colmap_format(colmap_dirpath)

    return dict(zip(img_fnames, wTi_list))


def compare_poses(baseline_dirpath: str, eval_dirpath: str, output_dirpath: str) -> None:
    """Compare the pose metrics between two reconstructions (Colmap format).

    Args:
        baseline_dirpath: Directory with baseline (reference) reconstruction.
        current_dirpath: Directory with reconstruction which needs evaluation.
        output_dirpath: Directory to save the metrics.
    """
    baseline_wTi_dict = load_poses(baseline_dirpath)
    current_wTi_dict = load_poses(eval_dirpath)

    common_fnames = baseline_wTi_dict.keys() & current_wTi_dict.keys()

    if args.use_pycolmap_alignment:
        current_reconstruction = pycolmap.Reconstruction(eval_dirpath)
        baselineScurrent = Similarity3(
            current_reconstruction.align_robust(
                list(baseline_wTi_dict.keys()),
                [wTi.translation() for wTi in baseline_wTi_dict.values()],
                min_common_images=3,
            ).matrix
        )
        current_wTi_dict = {fname: baselineScurrent.transformFrom(wTi) for fname, wTi in current_wTi_dict.items()}

        aRb = baselineScurrent.rotation().matrix()
        atb = baselineScurrent.translation()
        rz, ry, rx = Rotation.from_matrix(aRb).as_euler("zyx", degrees=True)
        logger.info("Sim(3) Rotation `aRb`: rz=%.2f deg., ry=%.2f deg., rx=%.2f deg.", rz, ry, rx)
        logger.info(f"Sim(3) Translation `atb`: [tx,ty,tz]={str(np.round(atb,2))}")
        logger.info("Sim(3) Scale `asb`: %.2f", float(baselineScurrent.scale()))

    logger.info(
        "Baseline: %d, current: %d , common: %d poses",
        len(baseline_wTi_dict),
        len(current_wTi_dict),
        len(common_fnames),
    )

    baseline_wTi_list: List[Pose3] = []
    current_wTi_list: List[Optional[Pose3]] = []
    for fname, wTi in baseline_wTi_dict.items():
        baseline_wTi_list.append(wTi)
        current_wTi_list.append(current_wTi_dict.get(fname))

    if not args.use_pycolmap_alignment:
        aSb = align.sim3_from_optional_Pose3s(baseline_wTi_list, current_wTi_list)
        current_wTi_list = transform.optional_Pose3s_with_sim3(current_wTi_list, aSb)

    i2Ri1_dict_gt, i2Ui1_dict_gt = metric_utils.get_all_relative_rotations_translations(baseline_wTi_list)

    wRi_aligned_list, wti_aligned_list = metric_utils.get_rotations_translations_from_poses(current_wTi_list)
    baseline_wRi_list, baseline_wti_list = metric_utils.get_rotations_translations_from_poses(baseline_wTi_list)

    metrics = []
    metrics.append(metric_utils.compute_rotation_angle_metric(wRi_aligned_list, baseline_wRi_list))
    metrics.append(metric_utils.compute_translation_distance_metric(wti_aligned_list, baseline_wti_list))
    metrics.append(metric_utils.compute_translation_angle_metric(baseline_wTi_list, current_wTi_list))
    relative_rotation_error_metric = metric_utils.compute_relative_rotation_angle_metric(
        i2Ri1_dict_gt, current_wTi_list
    )
    metrics.append(relative_rotation_error_metric)
    relative_translation_error_metric = metric_utils.compute_relative_translation_angle_metric(
        i2Ui1_dict_gt, current_wTi_list
    )
    metrics.append(relative_translation_error_metric)

    rotation_angular_errors = relative_rotation_error_metric._data
    translation_angular_errors = relative_translation_error_metric._data
    metrics.extend(
        metric_utils.compute_pose_auc_metric(
            rotation_angular_errors, translation_angular_errors, save_dir=output_dirpath
        )
    )

    ba_pose_metrics = GtsfmMetricsGroup(name="ba_pose_error_metrics", metrics=metrics)

    save_metrics_reports([ba_pose_metrics], metrics_path=output_dirpath)


if __name__ == "__main__":
    # Compare two reconstructions (in Colmap's output format). Right now, we just compare the poses.

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to directory containing benchmark artifacts for the baseline (reference).",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to directory containing benchmark artifacts for the current.",
    )
    parser.add_argument("--output", required=True, help="Output for the json file for pose metrics")
    parser.add_argument(
        "--use_pycolmap_alignment", action="store_true", help="Use Pycolmap to align cameras between two reconstruction"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    compare_poses(args.baseline, args.current, args.output)
