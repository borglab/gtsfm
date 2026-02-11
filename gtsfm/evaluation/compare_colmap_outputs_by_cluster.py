"""Compare COLMAP reconstructions using image-name alignment.

This script walks a results tree, finds cluster reconstructions under a given subfolder
name (default: "vggt"), and evaluates camera pose quality against a COLMAP baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gtsam import Pose3, Rot3

import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.utils import align

logger = logger_utils.get_logger()


def _read_images_txt_with_names(images_txt: Path) -> Dict[str, Pose3]:
    """Read poses from COLMAP images.txt keyed by image NAME."""
    if not images_txt.exists():
        raise FileNotFoundError(f"{images_txt} does not exist.")

    with images_txt.open("r") as f:
        lines = f.readlines()

    poses_by_name: Dict[str, Pose3] = {}
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        _image_id_str, qw, qx, qy, qz, tx, ty, tz, _camera_id = parts[:9]
        img_fname = " ".join(parts[9:])
        iRw = Rot3(float(qw), float(qx), float(qy), float(qz))
        wTi = Pose3(iRw, np.array([tx, ty, tz], dtype=np.float64)).inverse()
        if not np.isfinite(wTi.translation()).all():
            logger.warning("Skipping non-finite pose for %s in %s", img_fname, images_txt)
            continue
        poses_by_name[img_fname] = wTi
    return poses_by_name


def _read_images_txt_with_names_and_cameras(images_txt: Path) -> Tuple[Dict[str, Pose3], Dict[str, int]]:
    """Read poses and camera ids from COLMAP images.txt keyed by image NAME."""
    if not images_txt.exists():
        raise FileNotFoundError(f"{images_txt} does not exist.")

    with images_txt.open("r") as f:
        lines = f.readlines()

    poses_by_name: Dict[str, Pose3] = {}
    camera_by_name: Dict[str, int] = {}
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        _image_id_str, qw, qx, qy, qz, tx, ty, tz, camera_id_str = parts[:9]
        img_fname = " ".join(parts[9:])
        iRw = Rot3(float(qw), float(qx), float(qy), float(qz))
        wTi = Pose3(iRw, np.array([tx, ty, tz], dtype=np.float64)).inverse()
        if not np.isfinite(wTi.translation()).all():
            logger.warning("Skipping non-finite pose for %s in %s", img_fname, images_txt)
            continue
        camera_id = int(camera_id_str)
        poses_by_name[img_fname] = wTi
        camera_by_name[img_fname] = camera_id
    return poses_by_name, camera_by_name


def _read_cameras_txt_with_ids(cameras_txt: Path) -> Dict[int, Dict[str, float]]:
    """Read camera intrinsics from COLMAP cameras.txt keyed by CAMERA_ID."""
    if not cameras_txt.exists():
        raise FileNotFoundError(f"{cameras_txt} does not exist.")

    with cameras_txt.open("r") as f:
        lines = f.readlines()

    cameras_by_id: Dict[int, Dict[str, float]] = {}
    for line in lines[3:]:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = list(map(float, parts[4:]))
        if model == "SIMPLE_PINHOLE":
            fx, cx, cy = params
            fy = fx
            k1 = 0.0
            k2 = 0.0
        elif model == "SIMPLE_RADIAL":
            fx, cx, cy, k1 = params
            k2 = 0.0
            fy = fx
        elif model == "RADIAL":
            fx, cx, cy, k1, k2 = params
            fy = fx
        elif model == "PINHOLE":
            fx, fy, cx, cy = params
            k1 = 0.0
            k2 = 0.0
        elif model == "OPENCV":
            fx, fy, cx, cy, k1, k2, _p1, _p2, *_rest = params
        elif model == "OPENCV_FISHEYE":
            fx, fy, cx, cy, k1, k2, *_rest = params
        else:
            logger.warning("Unsupported camera model %s; skipping camera_id=%d", model, camera_id)
            continue
        cameras_by_id[camera_id] = {
            "model": model,
            "width": float(width),
            "height": float(height),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": k1,
            "k2": k2,
        }
    return cameras_by_id


def _find_cluster_recon_dirs(root: Path, recon_name: str) -> Iterable[Path]:
    """Yield directories that match the recon_name and contain images.txt."""
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) != recon_name:
            continue
        if "images.txt" in filenames:
            yield Path(dirpath)


def _build_pose_lists(
    baseline_poses: Dict[str, Pose3],
    current_poses: Dict[str, Pose3],
    cluster_label: str,
) -> Tuple[List[str], List[Pose3], List[Pose3]]:
    """Align poses by image NAME and return matched pose lists."""
    common_names = sorted(set(baseline_poses.keys()) & set(current_poses.keys()))
    if not common_names:
        missing_in_baseline = sorted(set(current_poses.keys()) - set(baseline_poses.keys()))
        missing_in_current = sorted(set(baseline_poses.keys()) - set(current_poses.keys()))
        if missing_in_baseline:
            logger.warning(
                "No common images for %s; missing in baseline (sample): %s",
                cluster_label,
                ", ".join(missing_in_baseline[:5]),
            )
        if missing_in_current:
            logger.warning(
                "No common images for %s; missing in current (sample): %s",
                cluster_label,
                ", ".join(missing_in_current[:5]),
            )
    else:
        logger.info("Common images for %s: %d", cluster_label, len(common_names))
    baseline_list = [baseline_poses[name] for name in common_names]
    current_list = [current_poses[name] for name in common_names]
    return common_names, baseline_list, current_list


def _compute_pose_metrics(baseline_list: List[Pose3], current_aligned_list: List[Pose3]) -> GtsfmMetricsGroup:
    """Compute the same pose metrics as compare_colmap_outputs, without plotting."""
    baseline_dict = {i: pose for i, pose in enumerate(baseline_list)}
    current_dict = {i: pose for i, pose in enumerate(current_aligned_list)}
    i2Ri1_dict_gt, i2Ui1_dict_gt = metric_utils.get_all_relative_rotations_translations(baseline_dict)
    wRi_aligned_dict, wti_aligned_dict = metric_utils.get_rotations_translations_from_poses(current_dict)
    baseline_wRi_dict, baseline_wti_dict = metric_utils.get_rotations_translations_from_poses(baseline_dict)

    metrics = []
    metrics.append(metric_utils.compute_rotation_angle_metric(wRi_aligned_dict, baseline_wRi_dict))
    metrics.append(metric_utils.compute_translation_distance_metric(wti_aligned_dict, baseline_wti_dict))
    metrics.append(metric_utils.compute_translation_angle_metric(baseline_dict, current_dict))
    relative_rotation_error_metric = metric_utils.compute_relative_rotation_angle_metric(
        i2Ri1_dict_gt, current_dict, store_full_data=True
    )
    metrics.append(relative_rotation_error_metric)
    relative_translation_error_metric = metric_utils.compute_relative_translation_angle_metric(
        i2Ui1_dict_gt, current_dict, store_full_data=True
    )
    metrics.append(relative_translation_error_metric)
    thresholds_deg = (1.0, 2.5, 5.0, 10.0, 20.0)
    if relative_rotation_error_metric.data is not None:
        rotation_angular_errors = np.asarray(relative_rotation_error_metric.data)
        rotation_auc_values = metric_utils.pose_auc(rotation_angular_errors, thresholds_deg)
        metrics.extend(
            [
                GtsfmMetric(f"rotation_auc_@{threshold}_deg", auc)
                for threshold, auc in zip(thresholds_deg, rotation_auc_values)
            ]
        )
    if relative_translation_error_metric.data is not None:
        translation_angular_errors = np.asarray(relative_translation_error_metric.data)
        translation_auc_values = metric_utils.pose_auc(translation_angular_errors, thresholds_deg)
        metrics.extend(
            [
                GtsfmMetric(f"translation_auc_@{threshold}_deg", auc)
                for threshold, auc in zip(thresholds_deg, translation_auc_values)
            ]
        )
    metrics.extend(
        metric_utils.compute_pose_auc_metric(
            relative_rotation_error_metric.data, relative_translation_error_metric.data, thresholds_deg=thresholds_deg
        )
    )

    return GtsfmMetricsGroup(name="ba_pose_error_metrics", metrics=metrics)


def _estimate_sim3_ransac(
    baseline_list: List[Pose3],
    current_list: List[Pose3],
    max_hypotheses: int,
    inlier_thresh: float,
    rng: np.random.Generator,
    cluster_label: str,
) -> align.Similarity3:
    """Estimate Sim(3) using simple RANSAC over camera centers with refit on inliers."""
    n_to_align = len(baseline_list)
    if n_to_align < 2:
        logger.warning("SIM(3) alignment uses at least 2 frames; Skipping for %s", cluster_label)
        return align.Similarity3(Rot3(), np.zeros(3), 1.0)

    baseline_centers = np.stack([pose.translation() for pose in baseline_list])
    current_centers = np.stack([pose.translation() for pose in current_list])
    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    best_mean_error = float("inf")
    best_aSb: Optional[align.Similarity3] = None

    for _ in range(max_hypotheses):
        sample_idx = rng.choice(n_to_align, size=2, replace=False)
        baseline_sample = {i: baseline_list[idx] for i, idx in enumerate(sample_idx)}
        current_sample = {i: current_list[idx] for i, idx in enumerate(sample_idx)}
        try:
            aSb_candidate = align.sim3_from_Pose3_maps(baseline_sample, current_sample)
        except Exception:
            continue
        transformed = np.stack([aSb_candidate.transformFrom(p) for p in current_centers])
        errors = np.linalg.norm(baseline_centers - transformed, axis=1)
        inliers = errors <= inlier_thresh
        count = int(np.count_nonzero(inliers))
        mean_error = float(errors[inliers].mean()) if count > 0 else float("inf")
        if count > best_count or (count == best_count and mean_error < best_mean_error):
            best_count = count
            best_mean_error = mean_error
            best_inliers = inliers
            best_aSb = aSb_candidate

    if best_aSb is None or best_inliers is None:
        logger.warning("Robust Sim3 failed; falling back to all-poses alignment for %s", cluster_label)
        baseline_dict = {i: pose for i, pose in enumerate(baseline_list)}
        current_dict = {i: pose for i, pose in enumerate(current_list)}
        return align.sim3_from_Pose3_maps(baseline_dict, current_dict)

    inlier_indices = np.where(best_inliers)[0]
    if len(inlier_indices) < 2:
        logger.warning(
            "Robust Sim3 inliers too few (%d/%d); using best hypothesis for %s",
            len(inlier_indices),
            n_to_align,
            cluster_label,
        )
        return best_aSb

    baseline_inliers = {i: baseline_list[idx] for i, idx in enumerate(inlier_indices)}
    current_inliers = {i: current_list[idx] for i, idx in enumerate(inlier_indices)}
    aSb_refit = align.sim3_from_Pose3_maps(baseline_inliers, current_inliers)
    logger.info(
        "Robust Sim3 for %s: inliers=%d/%d, thresh=%.3f",
        cluster_label,
        len(inlier_indices),
        n_to_align,
        inlier_thresh,
    )
    return aSb_refit


def _align_poses(
    baseline_list: List[Pose3],
    current_list: List[Pose3],
    use_ransac: bool,
    max_hypotheses: int,
    inlier_thresh: float,
    rng: np.random.Generator,
    cluster_label: str,
) -> Tuple[List[Pose3], align.Similarity3]:
    """Align current poses to baseline using Sim(3), optionally with RANSAC+refit."""
    baseline_dict = {i: pose for i, pose in enumerate(baseline_list)}
    current_dict = {i: pose for i, pose in enumerate(current_list)}
    if use_ransac:
        aSb = _estimate_sim3_ransac(baseline_list, current_list, max_hypotheses, inlier_thresh, rng, cluster_label)
    else:
        aSb = align.sim3_from_Pose3_maps(baseline_dict, current_dict)
    current_aligned_list = [aSb.transformFrom(pose) for pose in current_list]
    return current_aligned_list, aSb


def _plot_camera_centers(
    baseline_list: List[Pose3],
    current_list: List[Pose3],
    output_path: Path,
    title: str,
) -> None:
    """Save a 3D scatter plot of baseline and current camera centers."""
    baseline_centers = np.stack([pose.translation() for pose in baseline_list])
    current_centers_list = [pose.translation() for pose in current_list]
    current_centers = np.stack(current_centers_list) if current_centers_list else np.empty((0, 3))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    if baseline_centers.size:
        center = baseline_centers.mean(axis=0)
        mean_radius = np.linalg.norm(baseline_centers - center, axis=1).mean()
        arrow_len = max(mean_radius * 0.15, 1e-3)
    else:
        arrow_len = 1.0

    for pose in baseline_list:
        origin = pose.transformFrom(np.array([0.0, 0.0, 0.0]))
        tip = pose.transformFrom(np.array([0.0, 0.0, arrow_len]))
        direction = tip - origin
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color="tab:blue",
            linewidth=0.5,
            arrow_length_ratio=0.2,
            alpha=0.6,
        )
    for pose in current_list:
        origin = pose.transformFrom(np.array([0.0, 0.0, 0.0]))
        tip = pose.transformFrom(np.array([0.0, 0.0, arrow_len]))
        direction = tip - origin
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color="tab:orange",
            linewidth=0.5,
            arrow_length_ratio=0.2,
            alpha=0.6,
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
    if title:
        wrapped_lines = []
        for line in title.splitlines():
            wrapped_lines.extend(textwrap.wrap(line, width=80) or [""])
        wrapped = "\n".join(wrapped_lines)
    else:
        wrapped = ""
    if wrapped:
        fig.suptitle(wrapped, fontsize=9, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def _summarize_pose_errors(
    baseline_list: List[Pose3],
    current_aligned_list: List[Pose3],
    cluster_label: str,
) -> None:
    """Log median/mean absolute pose errors after alignment."""
    if not baseline_list or not current_aligned_list:
        return
    rot_errors_deg: List[float] = []
    trans_errors: List[float] = []
    for baseline_pose, current_pose in zip(baseline_list, current_aligned_list):
        rel = baseline_pose.between(current_pose)
        rot_vec = Rot3.Logmap(rel.rotation())
        rot_errors_deg.append(float(np.rad2deg(np.linalg.norm(rot_vec))))
        trans_errors.append(float(np.linalg.norm(rel.translation())))
    logger.info(
        "Pose errors for %s: rot_deg median=%.3f mean=%.3f; trans median=%.3f mean=%.3f",
        cluster_label,
        float(np.median(rot_errors_deg)),
        float(np.mean(rot_errors_deg)),
        float(np.median(trans_errors)),
        float(np.mean(trans_errors)),
    )


def _plot_pose_auc_boxplot(auc_values_by_label: Dict[str, List[float]], output_path: Path, title: str) -> None:
    """Save box plots for AUC metrics across all clusters."""
    preferred_order = ["@1.0_deg", "@2.5_deg", "@5.0_deg", "@10.0_deg", "@20.0_deg"]
    labels = [label for label in preferred_order if auc_values_by_label.get(label)]
    if not labels:
        labels = sorted(auc_values_by_label.keys())
    data = [auc_values_by_label[label] for label in labels if auc_values_by_label.get(label)]
    labels = [label for label in labels if auc_values_by_label.get(label)]
    if not data:
        return

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel("AUC")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    stats_lines = []
    for label, values in zip(labels, data):
        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        stats_lines.append(f"{label}: mean={mean_val:.3f}, med={median_val:.3f}")
    if stats_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def _plot_pose_auc_vs_input_images(
    auc_by_label_and_count: Dict[str, List[Tuple[int, float]]],
    output_path: Path,
) -> None:
    """Plot pose AUC at each threshold vs. input image count across clusters."""
    preferred_order = ["@1.0_deg", "@2.5_deg", "@5.0_deg", "@10.0_deg", "@20.0_deg"]
    labels = [label for label in preferred_order if auc_by_label_and_count.get(label)]
    if not labels:
        labels = sorted(auc_by_label_and_count.keys())

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    for label in labels:
        pairs = auc_by_label_and_count.get(label, [])
        if not pairs:
            continue
        pairs_sorted = sorted(pairs, key=lambda pair: pair[0])
        x_vals = [pair[0] for pair in pairs_sorted]
        y_vals = [pair[1] for pair in pairs_sorted]
        ax.plot(x_vals, y_vals, marker="o", linewidth=1.0, markersize=4, alpha=0.85, label=label)

    ax.set_title("Pose AUC vs input images (all clusters)")
    ax.set_xlabel("input images (current count)")
    ax.set_ylabel("AUC")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def _plot_intrinsics_deltas_boxplot(deltas: Dict[str, List[float]], output_path: Path, title: str) -> None:
    """Save box plots for normalized intrinsics deltas for a cluster."""
    labels = ["delta_fx_norm", "delta_fy_norm", "delta_cx_norm", "delta_cy_norm"]
    data = [deltas.get(label, []) for label in labels]
    if not any(data):
        return

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel("normalized by baseline value")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    stats_lines = []
    for label, values in zip(labels, data):
        if not values:
            continue
        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        stats_lines.append(f"{label}: mean={mean_val:.3f}, med={median_val:.3f}")
    if stats_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def _plot_fov_deltas_boxplot(deltas: Dict[str, List[float]], output_path: Path, title: str) -> None:
    """Save box plots for FOV deltas (degrees) for a cluster."""
    labels = ["delta_fovx_deg", "delta_fovy_deg"]
    data = [deltas.get(label, []) for label in labels]
    if not any(data):
        return

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel("degrees")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    stats_lines = []
    for label, values in zip(labels, data):
        if not values:
            continue
        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        stats_lines.append(f"{label}: mean={mean_val:.3f}, med={median_val:.3f}")
    if stats_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def _print_metrics(label: str, metrics_group: GtsfmMetricsGroup) -> None:
    logger.info("=== %s ===", label)
    for metric in metrics_group.metrics:
        if metric.dim == 0:
            value = "" if metric.data is None else f"{float(metric.data):.6f}"
            logger.info("%s: %s", metric.name, value)
        else:
            logger.info("%s: %s", metric.name, json.dumps(metric.summary, sort_keys=True))


def _format_auc(metrics_group: GtsfmMetricsGroup, prefix: str) -> str:
    auc_parts = []
    prefix_token = f"{prefix}_@"
    for metric in metrics_group.metrics:
        if not metric.name.startswith(prefix_token):
            continue
        if metric.data is None:
            continue
        try:
            value = float(metric.data)
        except (TypeError, ValueError):
            continue
        suffix = metric.name.replace(f"{prefix}_", "")
        auc_parts.append(f"{suffix}={value:.3f}")
    return ", ".join(auc_parts)


def export_metrics_group_to_csv(
    metrics_group: GtsfmMetricsGroup,
    cluster_label: str,
    baseline_count: int,
    current_count: int,
    common_count: int,
    output_path: Path,
    rows: List[Dict[str, str]],
) -> None:
    """Append metrics for a cluster into a shared CSV row list."""
    auc_values: List[float] = []
    for metric in metrics_group.metrics:
        if metric.dim == 0:
            value = "" if metric.data is None else f"{float(metric.data):.6f}"
            if metric.name.startswith("pose_auc_@") and metric.data is not None:
                try:
                    auc_values.append(float(metric.data))
                except (TypeError, ValueError):
                    pass
        else:
            value = json.dumps(metric.summary, sort_keys=True)
        rows.append(
            {
                "cluster": cluster_label,
                "baseline_count": str(baseline_count),
                "current_count": str(current_count),
                "common_count": str(common_count),
                "metric_name": metric.name,
                "value": value,
            }
        )
    if auc_values:
        rows.append(
            {
                "cluster": cluster_label,
                "baseline_count": str(baseline_count),
                "current_count": str(current_count),
                "common_count": str(common_count),
                "metric_name": "pose_auc_avg",
                "value": f"{float(np.mean(auc_values)):.6f}",
            }
        )

    if output_path.exists() and output_path.stat().st_size > 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["cluster", "baseline_count", "current_count", "common_count", "metric_name", "value"],
        )
        writer.writeheader()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline COLMAP directory.")
    parser.add_argument("--root", required=True, help="Root directory to traverse for cluster reconstructions.")
    parser.add_argument("--recon_name", default="vggt", help="Subdirectory name for reconstructions.")
    parser.add_argument(
        "--csv_output",
        default=None,
        help="Optional path to a single CSV file for all cluster metrics.",
    )
    parser.add_argument(
        "--fig_output_dir",
        default=None,
        help="Optional directory to save per-cluster camera_centers.png plots.",
    )
    parser.add_argument(
        "--robust_sim3",
        action="store_true",
        default=False,
        help="Use simple RANSAC+refit for Sim(3) alignment.",
    )
    parser.add_argument(
        "--robust_sim3_max_hypotheses",
        type=int,
        default=200,
        help="Max RANSAC hypotheses for robust Sim(3) alignment.",
    )
    parser.add_argument(
        "--robust_sim3_inlier_thresh",
        type=float,
        default=0.1,
        help="Inlier threshold on camera-center error for robust Sim(3) alignment.",
    )
    parser.add_argument(
        "--robust_sim3_seed",
        type=int,
        default=0,
        help="Random seed for robust Sim(3) alignment.",
    )
    args = parser.parse_args()

    baseline_images = Path(args.baseline) / "images.txt"
    baseline_cameras_txt = Path(args.baseline) / "cameras.txt"
    baseline_poses, baseline_camera_by_name = _read_images_txt_with_names_and_cameras(baseline_images)
    baseline_cameras = _read_cameras_txt_with_ids(baseline_cameras_txt)
    fig_output_dir = Path(args.fig_output_dir) if args.fig_output_dir else None
    if fig_output_dir is None and args.csv_output:
        fig_output_dir = Path(args.csv_output).parent / "cluster_camera_centers"

    root = Path(args.root)
    recon_dirs = sorted(_find_cluster_recon_dirs(root, args.recon_name))
    if not recon_dirs:
        raise FileNotFoundError(f"No reconstructions named '{args.recon_name}' with images.txt under {root}")

    logger.info("Found %d reconstructions under %s", len(recon_dirs), root)

    csv_rows: List[Dict[str, str]] = []
    all_pose_auc_values: Dict[str, List[float]] = {}
    all_pose_auc_by_label_and_count: Dict[str, List[Tuple[int, float]]] = {}
    all_rotation_auc_values: Dict[str, List[float]] = {}
    all_translation_auc_values: Dict[str, List[float]] = {}
    all_intrinsics_deltas: Dict[str, List[float]] = {
        "delta_fx_norm": [],
        "delta_fy_norm": [],
        "delta_cx_norm": [],
        "delta_cy_norm": [],
    }
    all_fov_deltas: Dict[str, List[float]] = {
        "delta_fovx_deg": [],
        "delta_fovy_deg": [],
    }
    rng = np.random.default_rng(args.robust_sim3_seed)
    for recon_dir in recon_dirs:
        current_images = recon_dir / "images.txt"
        current_cameras_txt = recon_dir / "cameras.txt"
        current_poses, current_camera_by_name = _read_images_txt_with_names_and_cameras(current_images)
        try:
            current_cameras = _read_cameras_txt_with_ids(current_cameras_txt)
        except FileNotFoundError:
            logger.warning("Missing cameras.txt for %s; skipping intrinsics comparison.", recon_dir)
            current_cameras = {}
        common_names, baseline_list, current_list = _build_pose_lists(
            baseline_poses, current_poses, cluster_label=str(recon_dir)
        )
        baseline_count = len(baseline_poses)
        current_count = len(current_poses)
        common_count = len(common_names)
        if len(common_names) < 2:
            logger.warning(
                "Skipping %s (baseline=%d, current=%d, common=%d)",
                recon_dir,
                baseline_count,
                current_count,
                common_count,
            )
            continue
        current_aligned_list, _aSb = _align_poses(
            baseline_list,
            current_list,
            use_ransac=args.robust_sim3,
            max_hypotheses=args.robust_sim3_max_hypotheses,
            inlier_thresh=args.robust_sim3_inlier_thresh,
            rng=rng,
            cluster_label=str(recon_dir),
        )
        metrics_group = _compute_pose_metrics(baseline_list, current_aligned_list)
        _summarize_pose_errors(baseline_list, current_aligned_list, str(recon_dir))
        intrinsics_deltas: Dict[str, List[float]] = {
            "delta_fx_norm": [],
            "delta_fy_norm": [],
            "delta_cx_norm": [],
            "delta_cy_norm": [],
        }
        fov_deltas: Dict[str, List[float]] = {
            "delta_fovx_deg": [],
            "delta_fovy_deg": [],
        }
        for name in common_names:
            base_cam_id = baseline_camera_by_name.get(name)
            curr_cam_id = current_camera_by_name.get(name)
            if base_cam_id is None or curr_cam_id is None:
                continue
            base = baseline_cameras.get(base_cam_id)
            curr = current_cameras.get(curr_cam_id)
            if base is None or curr is None:
                continue
            base_w, base_h = base["width"], base["height"]
            curr_w, curr_h = curr["width"], curr["height"]
            if curr_w > 0 and curr_h > 0 and (base_w != curr_w or base_h != curr_h):
                sx = base_w / curr_w
                sy = base_h / curr_h
                curr_fx = curr["fx"] * sx
                curr_fy = curr["fy"] * sy
                curr_cx = curr["cx"] * sx
                curr_cy = curr["cy"] * sy
            else:
                curr_fx = curr["fx"]
                curr_fy = curr["fy"]
                curr_cx = curr["cx"]
                curr_cy = curr["cy"]
            if base_w > 0 and base_h > 0:
                if base["fx"] != 0:
                    intrinsics_deltas["delta_fx_norm"].append(abs(base["fx"] - curr_fx) / abs(base["fx"]))
                if base["fy"] != 0:
                    intrinsics_deltas["delta_fy_norm"].append(abs(base["fy"] - curr_fy) / abs(base["fy"]))
                if base["cx"] != 0:
                    intrinsics_deltas["delta_cx_norm"].append(abs(base["cx"] - curr_cx) / abs(base["cx"]))
                if base["cy"] != 0:
                    intrinsics_deltas["delta_cy_norm"].append(abs(base["cy"] - curr_cy) / abs(base["cy"]))
            base_fovx = 2.0 * np.degrees(np.arctan(base_w / (2.0 * base["fx"])))
            base_fovy = 2.0 * np.degrees(np.arctan(base_h / (2.0 * base["fy"])))
            curr_fovx = 2.0 * np.degrees(np.arctan(base_w / (2.0 * curr_fx)))
            curr_fovy = 2.0 * np.degrees(np.arctan(base_h / (2.0 * curr_fy)))
            fov_deltas["delta_fovx_deg"].append(abs(base_fovx - curr_fovx))
            fov_deltas["delta_fovy_deg"].append(abs(base_fovy - curr_fovy))
        for key, values in intrinsics_deltas.items():
            all_intrinsics_deltas[key].extend(values)
        for key, values in fov_deltas.items():
            all_fov_deltas[key].extend(values)
        if args.csv_output:
            export_metrics_group_to_csv(
                metrics_group,
                cluster_label=str(recon_dir),
                baseline_count=baseline_count,
                current_count=current_count,
                common_count=common_count,
                output_path=Path(args.csv_output),
                rows=csv_rows,
            )
        else:
            _print_metrics(str(recon_dir), metrics_group)
        if fig_output_dir is not None:
            safe_name = str(recon_dir).replace(os.sep, "__")
            plot_path = fig_output_dir / f"{safe_name}_camera_centers.png"
            pose_auc_text = _format_auc(metrics_group, "pose_auc")
            rotation_auc_text = _format_auc(metrics_group, "rotation_auc")
            translation_auc_text = _format_auc(metrics_group, "translation_auc")
            title_lines = [f"{recon_dir}"]
            if pose_auc_text:
                title_lines.append(f"Pose AUC: {pose_auc_text}")
            if rotation_auc_text:
                title_lines.append(f"Rotation AUC: {rotation_auc_text}")
            if translation_auc_text:
                title_lines.append(f"Translation AUC: {translation_auc_text}")
            title = "\n".join(title_lines)
            _plot_camera_centers(baseline_list, current_aligned_list, plot_path, title)
        # Intrinsics stats are annotated in the plot; no terminal logging.
        for metric in metrics_group.metrics:
            if metric.name.startswith("pose_auc_@") and metric.data is not None:
                try:
                    value = float(metric.data)
                except (TypeError, ValueError):
                    continue
                label = metric.name.replace("pose_auc_", "")
                all_pose_auc_values.setdefault(label, []).append(value)
                all_pose_auc_by_label_and_count.setdefault(label, []).append((current_count, value))
            elif metric.name.startswith("rotation_auc_@") and metric.data is not None:
                try:
                    value = float(metric.data)
                except (TypeError, ValueError):
                    continue
                label = metric.name.replace("rotation_auc_", "")
                all_rotation_auc_values.setdefault(label, []).append(value)
            elif metric.name.startswith("translation_auc_@") and metric.data is not None:
                try:
                    value = float(metric.data)
                except (TypeError, ValueError):
                    continue
                label = metric.name.replace("translation_auc_", "")
                all_translation_auc_values.setdefault(label, []).append(value)

    if args.csv_output and csv_rows:
        output_path = Path(args.csv_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["cluster", "baseline_count", "current_count", "common_count", "metric_name", "value"],
            )
            writer.writerows(csv_rows)
    if fig_output_dir is not None and all_pose_auc_values:
        auc_plot_path = fig_output_dir / "pose_auc_boxplot_all_clusters.png"
        _plot_pose_auc_boxplot(all_pose_auc_values, auc_plot_path, "Pose AUC by threshold (all clusters)")
    if fig_output_dir is not None and all_pose_auc_by_label_and_count:
        auc_vs_images_plot_path = fig_output_dir / "pose_auc_vs_input_images.png"
        _plot_pose_auc_vs_input_images(all_pose_auc_by_label_and_count, auc_vs_images_plot_path)
    if fig_output_dir is not None and all_rotation_auc_values:
        rotation_auc_plot_path = fig_output_dir / "rotation_auc_boxplot_all_clusters.png"
        _plot_pose_auc_boxplot(
            all_rotation_auc_values, rotation_auc_plot_path, "Rotation AUC by threshold (all clusters)"
        )
    if fig_output_dir is not None and all_translation_auc_values:
        translation_auc_plot_path = fig_output_dir / "translation_auc_boxplot_all_clusters.png"
        _plot_pose_auc_boxplot(
            all_translation_auc_values,
            translation_auc_plot_path,
            "Translation AUC by threshold (all clusters)",
        )
    if fig_output_dir is not None and any(all_intrinsics_deltas.values()):
        intrinsics_plot_path = fig_output_dir / "intrinsics_deltas_all_clusters.png"
        _plot_intrinsics_deltas_boxplot(
            all_intrinsics_deltas,
            intrinsics_plot_path,
            "Intrinsics Δ (normalized, all clusters)",
        )
    if fig_output_dir is not None and any(all_fov_deltas.values()):
        fov_plot_path = fig_output_dir / "fov_deltas_all_clusters.png"
        _plot_fov_deltas_boxplot(
            all_fov_deltas,
            fov_plot_path,
            "FOV Δ (degrees, all clusters)",
        )


if __name__ == "__main__":
    main()
