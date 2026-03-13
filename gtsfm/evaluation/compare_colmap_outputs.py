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
from thirdparty.colmap.scripts.python import read_write_model

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


def _image_candidates(image_name: str, root_dirs: List[str]) -> List[str]:
    """Generate common candidate paths for an image under a list of root directories."""
    base_name = os.path.basename(image_name)
    candidates: List[str] = []
    for root_dir in root_dirs:
        if root_dir == "":
            continue
        candidates.extend(
            [
                os.path.join(root_dir, image_name),
                os.path.join(root_dir, base_name),
                os.path.join(root_dir, "images", image_name),
                os.path.join(root_dir, "images", base_name),
            ]
        )
    return list(dict.fromkeys([path for path in candidates if path]))


def _find_image_path(image_name: str, root_dirs: List[str]) -> Optional[str]:
    """Find an image file path from candidate roots."""
    if os.path.isabs(image_name) and os.path.exists(image_name):
        return image_name

    for path in _image_candidates(image_name, root_dirs):
        if os.path.exists(path):
            return path

    return None


def _find_images_file_in_reconstruction(model_dir: str) -> Optional[str]:
    """Find COLMAP images.txt or images.bin under a reconstruction directory."""
    candidates = [
        model_dir,
        os.path.join(model_dir, "sparse"),
        os.path.join(model_dir, "sparse/0"),
        os.path.join(model_dir, "0"),
    ]
    for base in candidates:
        for fname in ("images.txt", "images.bin"):
            candidate = os.path.join(base, fname)
            if os.path.isfile(candidate):
                return candidate
    for root, _dirs, files in os.walk(model_dir):
        if "images.txt" in files:
            return os.path.join(root, "images.txt")
        if "images.bin" in files:
            return os.path.join(root, "images.bin")
    return None


def _get_current_image_measurement_counts(current_recon_dir: str) -> Dict[str, int]:
    """Read number of point observations per image from the current COLMAP reconstruction."""
    images_file = _find_images_file_in_reconstruction(current_recon_dir)
    if images_file is None:
        logger.warning("No images.txt/images.bin found in %s; cannot read measurement counts.", current_recon_dir)
        return {}

    try:
        if images_file.endswith(".bin"):
            images = read_write_model.read_images_binary(images_file)
        else:
            images = read_write_model.read_images_text(images_file)
    except Exception as e:
        logger.warning("Failed to read %s (%s): %s", images_file, type(e).__name__, str(e))
        return {}

    counts = {img.name: len(img.xys) for _, img in images.items()}
    logger.info("Loaded %d images with measurement counts from %s.", len(counts), images_file)
    return counts


def _save_image_with_error_overlay(
    src_path: str, dst_path: str, error: float, metric_name: str, num_measurements: Optional[object] = None
) -> None:
    """Save an image copy with the metric error text drawn on top."""
    image = plt.imread(src_path)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    ax.axis("off")
    measurement_text = "N/A" if num_measurements is None else str(num_measurements)
    ax.text(
        0.02,
        0.02,
        f"{metric_name}: {float(error):.4f}\nnum_measurements: {measurement_text}",
        color="yellow",
        fontsize=14,
        weight="bold",
        transform=ax.transAxes,
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3.0),
    )
    fig.savefig(dst_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _get_measurement_count_for_image(
    image_name: str, image_measurement_counts: Optional[Dict[str, int]]
) -> Optional[int]:
    """Get per-image measurement count by matching full name or basename."""
    if image_measurement_counts is None:
        return None
    base_name = os.path.basename(image_name)
    return image_measurement_counts.get(image_name, image_measurement_counts.get(base_name))


def _plot_error_vs_measurements(
    metric: metric_utils.GtsfmMetric,
    image_names: List[str],
    image_measurement_counts: Optional[Dict[str, int]],
    output_dirpath: str,
    metric_folder: str,
) -> None:
    """Save a scatter plot of metric error versus per-image measurement count."""
    if metric.data is None:
        logger.warning("Skipping error-vs-measurements plot for metric `%s`: no full data.", metric.name)
        return

    errors = np.asarray(metric.data, dtype=np.float32)
    if errors.size != len(image_names):
        logger.warning(
            "Skipping error-vs-measurements plot for metric `%s`: mismatch between errors (%d) and image names (%d).",
            metric.name,
            int(errors.size),
            len(image_names),
        )
        return

    valid = np.isfinite(errors)
    valid_errors = errors[valid]
    if valid_errors.size == 0:
        logger.warning("Skipping error-vs-measurements plot for metric `%s`: no finite errors.", metric.name)
        return

    valid_names = np.array(image_names, dtype=object)[valid]
    counts = []
    for name in valid_names:
        count = _get_measurement_count_for_image(str(name), image_measurement_counts)
        counts.append(np.nan if count is None else float(count))

    counts_np = np.asarray(counts, dtype=np.float32)
    valid_count_mask = np.isfinite(counts_np)
    if not np.any(valid_count_mask):
        logger.warning(
            "Skipping error-vs-measurements plot for metric `%s`: no numeric measurement counts.", metric.name
        )
        return

    metric_output_dir = os.path.join(output_dirpath, metric_folder)
    os.makedirs(metric_output_dir, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(counts_np[valid_count_mask], valid_errors[valid_count_mask], s=12, alpha=0.55)
    ax.set_title(f"{metric.name}: error vs num_measurements")
    ax.set_xlabel("num_measurements")
    ax.set_ylabel(metric.name)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(metric_output_dir, f"{metric_folder}_error_vs_measurements.png"), dpi=250)
    plt.close(fig)


def _export_ranked_images(
    metric: metric_utils.GtsfmMetric,
    image_names: List[str],
    image_roots: List[str],
    output_dirpath: str,
    metric_folder: str,
    image_measurement_counts: Optional[Dict[str, int]] = None,
) -> None:
    """Export images sorted by metric value in descending order.

    The file name format is {rank}_{image_name}.
    """
    if metric.data is None:
        logger.warning("Skipping image export for metric `%s`: no full data.", metric.name)
        return

    errors = np.asarray(metric.data, dtype=np.float32)
    if errors.size != len(image_names):
        logger.warning(
            "Skipping image export for metric `%s`: mismatch between errors (%d) and image names (%d).",
            metric.name,
            int(errors.size),
            len(image_names),
        )
        return

    valid = np.isfinite(errors)
    valid_errors = errors[valid]
    sorted_indices = np.argsort(valid_errors)[::-1]
    valid_names = np.array(image_names, dtype=object)[valid]

    output_metric_dir = os.path.join(output_dirpath, metric_folder)
    os.makedirs(output_metric_dir, exist_ok=True)

    missing_count_for_images = []

    for rank, sorted_idx in enumerate(sorted_indices):
        image_name = str(valid_names[sorted_idx])
        src = _find_image_path(image_name, image_roots)
        if src is None:
            logger.warning("Could not find image file for %s.", image_name)
            continue
        dst = os.path.join(output_metric_dir, f"{rank}_{os.path.basename(image_name)}")
        error = float(valid_errors[sorted_idx])
        num_measurements = "N/A"
        if image_measurement_counts is not None:
            resolved_measurements = _get_measurement_count_for_image(image_name, image_measurement_counts)
            if resolved_measurements is not None:
                num_measurements = resolved_measurements
            else:
                missing_count_for_images.append(image_name)
        _save_image_with_error_overlay(src, dst, error, metric.name, num_measurements=num_measurements)

    if missing_count_for_images:
        sample = ", ".join(missing_count_for_images[:5])
        logger.warning(
            "No measurement count for %d images in metric `%s` (sample: %s).",
            len(missing_count_for_images),
            metric.name,
            sample,
        )


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
        aSb = align.sim3_from_Pose3_maps_robust(baseline_wTi_dict, current_wTi_dict)
        current_wTi_list = transform.optional_Pose3s_with_sim3(aSb, current_wTi_list)
        current_wTi_dict = {fname: aSb.transformFrom(pose) for fname, pose in current_wTi_dict.items()}

    i2Ri1_dict_gt, i2Ui1_dict_gt = metric_utils.get_all_relative_rotations_translations(baseline_wTi_dict)

    wRi_aligned_dict, wti_aligned_dict = metric_utils.get_rotations_translations_from_poses(current_wTi_dict)
    baseline_wRi_dict, baseline_wti_dict = metric_utils.get_rotations_translations_from_poses(baseline_wTi_dict)

    metrics = []
    metrics.append(
        metric_utils.compute_rotation_angle_metric(wRi_aligned_dict, baseline_wRi_dict, store_full_data=True)
    )
    metrics.append(
        metric_utils.compute_translation_distance_metric(wti_aligned_dict, baseline_wti_dict, store_full_data=True)
    )
    metrics.append(metric_utils.compute_translation_angle_metric(baseline_wTi_dict, current_wTi_dict))
    relative_rotation_error_metric = metric_utils.compute_relative_rotation_angle_metric(
        i2Ri1_dict_gt, current_wTi_dict, store_full_data=True
    )
    metrics.append(relative_rotation_error_metric)
    relative_translation_error_metric = metric_utils.compute_relative_translation_angle_metric(
        i2Ui1_dict_gt, current_wTi_dict, store_full_data=True
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

    image_roots = [
        baseline_dirpath,
        eval_dirpath,
        os.path.join(baseline_dirpath, "images"),
        os.path.join(eval_dirpath, "images"),
    ]
    if args.image_root is not None:
        image_roots.insert(0, args.image_root)
    image_roots = list(dict.fromkeys([root for root in image_roots if root]))

    image_names = sorted(baseline_wRi_dict.keys())
    rotation_angle_metric = metrics[0]
    translation_distance_metric = metrics[1]
    current_image_measurement_counts = _get_current_image_measurement_counts(eval_dirpath)
    if current_image_measurement_counts:
        valid_count_values = [
            _get_measurement_count_for_image(name, current_image_measurement_counts)
            for name in image_names
            if _get_measurement_count_for_image(name, current_image_measurement_counts) is not None
        ]
        if valid_count_values:
            logger.info(
                "Current reconstruction measurement stats: n=%d, min=%d, max=%d, mean=%.2f",
                len(valid_count_values),
                int(min(valid_count_values)),
                int(max(valid_count_values)),
                float(np.mean(valid_count_values)),
            )
    _export_ranked_images(
        rotation_angle_metric,
        image_names,
        image_roots,
        output_dirpath,
        "rotation_angle_metric",
        image_measurement_counts=current_image_measurement_counts,
    )
    _plot_error_vs_measurements(
        rotation_angle_metric,
        image_names,
        current_image_measurement_counts,
        output_dirpath,
        "rotation_angle_metric",
    )
    _export_ranked_images(
        translation_distance_metric,
        image_names,
        image_roots,
        output_dirpath,
        "translation_distance_metric",
        image_measurement_counts=current_image_measurement_counts,
    )
    _plot_error_vs_measurements(
        translation_distance_metric,
        image_names,
        current_image_measurement_counts,
        output_dirpath,
        "translation_distance_metric",
    )

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
    parser.add_argument(
        "--image_root",
        default=None,
        help="Optional directory containing source images. If provided, script copies images into metric folders.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ba_pose_metrics = compare_poses(args.baseline, args.current, args.output)
    export_metrics_group_to_csv(ba_pose_metrics, os.path.join(args.output, f"{ba_pose_metrics.name}.csv"))
