"""Script to compare two reconstructions in Colmap's output format.

Authors: Ayush Baid, Xinan Zhang
"""

import argparse
import csv
import json
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pycolmap
from gtsam import Point3, Pose3, Rot3, Similarity3
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


def align_with_colmap(
    baseline_wTi_dict: Dict[str, Pose3], current_wTi_dict: Dict[str, Pose3]
) -> Tuple[Similarity3, Dict[str, Pose3]]:
    """Align pose dictionaries using COLMAP's similarity estimation on camera centers."""
    common_fnames = sorted(baseline_wTi_dict.keys() & current_wTi_dict.keys())
    if len(common_fnames) < 2:
        raise RuntimeError("Need at least two overlapping cameras for COLMAP-based alignment.")

    baseline_centers = np.stack([baseline_wTi_dict[fname].translation() for fname in common_fnames])
    current_centers = np.stack([current_wTi_dict[fname].translation() for fname in common_fnames])

    ransac_opts = pycolmap.RANSACOptions()
    ransac_opts.max_error = 0.1
    sim = pycolmap.estimate_sim3d_robust(current_centers, baseline_centers, ransac_opts)

    quat_xyzw = sim["tgt_from_src"].rotation.quat
    aRb = Rot3.Quaternion(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
    atb = Point3(sim["tgt_from_src"].translation)
    aSb = Similarity3(aRb, atb, sim["tgt_from_src"].scale)

    aligned_dict = {fname: aSb.transformFrom(pose) for fname, pose in current_wTi_dict.items()}
    return aSb, aligned_dict


def plot_camera_centers(
    baseline_wTi_list: List[Pose3],
    current_wTi_list: List[Pose3],
    output_dirpath: str,
    title: Optional[str] = None,
) -> None:
    """Save a 3D scatter plot of baseline and current camera centers."""
    baseline_centers = np.stack([pose.translation() for pose in baseline_wTi_list])
    current_centers_list = [pose.translation() for pose in current_wTi_list]
    current_centers = np.stack(current_centers_list) if current_centers_list else np.empty((0, 3))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    if baseline_centers.size:
        center = baseline_centers.mean(axis=0)
        mean_radius = np.linalg.norm(baseline_centers - center, axis=1).mean()
        arrow_len = max(mean_radius * 0.15, 1e-3)
    else:
        arrow_len = 1.0

    for pose in baseline_wTi_list:
        origin = pose.transformFrom(Point3(0.0, 0.0, 0.0))
        tip = pose.transformFrom(Point3(0.0, 0.0, arrow_len))
        direction = tip - origin
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color="tab:blue", linewidth=0.5, arrow_length_ratio=0.2, alpha=0.6
        )
    for pose in current_wTi_list:
        origin = pose.transformFrom(Point3(0.0, 0.0, 0.0))
        tip = pose.transformFrom(Point3(0.0, 0.0, arrow_len))
        direction = tip - origin
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color="tab:orange", linewidth=0.5, arrow_length_ratio=0.2, alpha=0.6
        )

    ax.scatter(
        baseline_centers[:, 0],
        baseline_centers[:, 1],
        baseline_centers[:, 2],
        s=10,
        c="tab:blue",
        label="baseline",
    )
    if current_centers.size:
        ax.scatter(
            current_centers[:, 0],
            current_centers[:, 1],
            current_centers[:, 2],
            s=10,
            c="tab:orange",
            label="current",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")
    wrapped = "\n".join(textwrap.wrap(title, width=80)) if title else ""
    if wrapped:
        fig.suptitle(wrapped, fontsize=9, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(output_dirpath, "camera_centers.png"), dpi=300)
    plt.close(fig)


def export_metrics_group_to_csv(metrics_group: GtsfmMetricsGroup, output_path: str) -> None:
    """Export a metrics group to a CSV file."""
    rows: List[Dict[str, str]] = []
    for metric in metrics_group.metrics:
        if metric.dim == 0:
            value = "" if metric.data is None else f"{float(metric.data):.6f}"
            rows.append({"metric_name": metric.name, "value": value})
        else:
            summary_json = json.dumps(metric.summary, sort_keys=True)
            rows.append({"metric_name": metric.name, "value": summary_json})

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["metric_name", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _format_pose_auc(metrics_group: GtsfmMetricsGroup) -> str:
    auc_parts = []
    for metric in metrics_group.metrics:
        if not metric.name.startswith("pose_auc_@"):
            continue
        if metric.data is None:
            continue
        try:
            value = float(metric.data)
        except (TypeError, ValueError):
            continue
        suffix = metric.name.replace("pose_auc_", "")
        auc_parts.append(f"{suffix}={value:.3f}")
    return ", ".join(auc_parts)


def compare_poses(baseline_dirpath: str, eval_dirpath: str, output_dirpath: str) -> GtsfmMetricsGroup:
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
        baselineScurrent, current_wTi_dict = align_with_colmap(baseline_wTi_dict, current_wTi_dict)
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
        current_wTi_list = transform.optional_Pose3s_with_sim3(aSb, current_wTi_list)
        current_wTi_dict = {fname: aSb.transformFrom(pose) for fname, pose in current_wTi_dict.items()}

    i2Ri1_dict_gt, i2Ui1_dict_gt = metric_utils.get_all_relative_rotations_translations(baseline_wTi_list)

    wRi_aligned_list, wti_aligned_list = metric_utils.get_rotations_translations_from_poses(current_wTi_list)
    baseline_wRi_list, baseline_wti_list = metric_utils.get_rotations_translations_from_poses(baseline_wTi_list)

    metrics = []
    metrics.append(metric_utils.compute_rotation_angle_metric(wRi_aligned_list, baseline_wRi_list))
    metrics.append(metric_utils.compute_translation_distance_metric(wti_aligned_list, baseline_wti_list))
    metrics.append(metric_utils.compute_translation_angle_metric(baseline_wTi_list, current_wTi_list))
    relative_rotation_error_metric = metric_utils.compute_relative_rotation_angle_metric(
        i2Ri1_dict_gt, current_wTi_list, store_full_data=True
    )
    metrics.append(relative_rotation_error_metric)
    relative_translation_error_metric = metric_utils.compute_relative_translation_angle_metric(
        i2Ui1_dict_gt, current_wTi_list, store_full_data=True
    )
    metrics.append(relative_translation_error_metric)

    rotation_angular_errors = relative_rotation_error_metric.data
    translation_angular_errors = relative_translation_error_metric.data
    metrics.extend(
        metric_utils.compute_pose_auc_metric(
            rotation_angular_errors, translation_angular_errors, save_dir=output_dirpath
        )
    )

    ba_pose_metrics = GtsfmMetricsGroup(name="ba_pose_error_metrics", metrics=metrics)

    auc_text = _format_pose_auc(ba_pose_metrics)
    title = eval_dirpath
    if auc_text:
        title = f"{title}\nPose AUC: {auc_text}"
    plot_camera_centers(baseline_wTi_list, list(current_wTi_dict.values()), output_dirpath, title=title)

    save_metrics_reports([ba_pose_metrics], metrics_path=output_dirpath)
    return ba_pose_metrics


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

    ba_pose_metrics = compare_poses(args.baseline, args.current, args.output)
    export_metrics_group_to_csv(ba_pose_metrics, os.path.join(args.output, f"{ba_pose_metrics.name}.csv"))

