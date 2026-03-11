"""Check track quality for COLMAP reconstructions under cluster folders.

This script:
1) Finds cluster model folders named `--model_name` under `--recon_root`.
2) Computes reprojection errors for all track measurements in each cluster.
3) Saves summary statistics (JSON + CSV).
4) Saves overlay visualizations for up to `--max_images` images per cluster.

Example:
    python pipeline/utils/check_tracks.py \
        --recon_root /nethome/xzhang979/nvme/gtsfm/pipeline/results/gerrard-hall/2-reconstruction/vggt_cluster_run/results \
        --images_root /nethome/xzhang979/nvme/gtsfm/benchmarks/gerrard-hall \
        --model_name vggt
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from gtsfm.common.gtsfm_data import GtsfmData


@dataclass
class MeasurementPair:
    """One 2D measurement and its reprojection for visualization/error calculation."""

    track_idx: int
    measured: np.ndarray
    reprojected: np.ndarray
    error: float


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and visualize track reprojection errors in cluster reconstructions.")
    parser.add_argument("--recon_root", type=str, required=True, help="Root directory containing per-cluster outputs.")
    parser.add_argument("--images_root", type=str, required=True, help="Root directory for source images.")
    parser.add_argument("--model_name", type=str, default="vggt", help="COLMAP model directory name per cluster.")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output root for reports/visualizations. Defaults to <recon_root>/check_tracks.",
    )
    parser.add_argument("--max_images", type=int, default=5, help="Maximum overlay images to save per cluster.")
    parser.add_argument(
        "--max_tracks_per_image",
        type=int,
        default=500,
        help="Maximum track measurements to draw per image (for readability/speed).",
    )
    parser.add_argument("--line_width", type=int, default=1, help="Line width in visualization.")
    parser.add_argument("--dot_radius", type=int, default=2, help="Dot radius in visualization.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for visualization sampling.")
    parser.add_argument(
        "--max_correspondences_per_pair",
        type=int,
        default=20,
        help="Maximum cross-image correspondence lines to draw for each adjacent image pair.",
    )
    parser.add_argument(
        "--corr_low_residual_only",
        action="store_true",
        help="For stitched correspondences, only keep points with low reprojection residual.",
    )
    parser.add_argument(
        "--corr_residual_thresh_px",
        type=float,
        default=None,
        help="Residual threshold (px) for --corr_low_residual_only. Defaults to cluster p95 if omitted.",
    )
    parser.add_argument("--hist_bins", type=int, default=80, help="Number of bins for reprojection histograms.")
    parser.add_argument(
        "--hist_clip_px",
        type=float,
        default=100.0,
        help="Clip histogram x-axis to this pixel value for readability.",
    )
    return parser.parse_args()


def _is_colmap_model_dir(model_dir: Path) -> bool:
    has_images = (model_dir / "images.txt").exists() or (model_dir / "images.bin").exists()
    has_points = (model_dir / "points3D.txt").exists() or (model_dir / "points3D.bin").exists()
    has_cameras = (model_dir / "cameras.txt").exists() or (model_dir / "cameras.bin").exists()
    return has_images and has_points and has_cameras


def _find_model_dirs(recon_root: Path, model_name: str) -> list[Path]:
    matches: list[Path] = []
    for path in recon_root.rglob(model_name):
        if path.is_dir() and _is_colmap_model_dir(path):
            matches.append(path)
    return sorted(matches)


def _parse_colmap_recon_sizes(model_dir: Path) -> dict[int, tuple[int, int]]:
    """Parse image_id -> (recon_h, recon_w) from COLMAP text model files, if available."""
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"
    if not cameras_txt.exists() or not images_txt.exists():
        return {}

    camera_hw: dict[int, tuple[int, int]] = {}
    for line in cameras_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            camera_id = int(parts[0])
            width = int(parts[2])
            height = int(parts[3])
        except ValueError:
            continue
        camera_hw[camera_id] = (height, width)

    image_hw: dict[int, tuple[int, int]] = {}
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 9:
            try:
                image_id = int(parts[0])
                camera_id = int(parts[8])
            except ValueError:
                i += 2
                continue
            if camera_id in camera_hw:
                image_hw[image_id] = camera_hw[camera_id]
        i += 2

    return image_hw


def _build_image_index(images_root: Path) -> dict[str, list[Path]]:
    """Index all images by lowercase basename to support flexible lookup."""
    index: dict[str, list[Path]] = {}
    for p in images_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = p.name.lower()
        index.setdefault(key, []).append(p)
    return index


def _get_image_path(image_name: str, images_root: Path, image_index: dict[str, list[Path]]) -> Optional[Path]:
    # 1) Try relative path from images_root as-is.
    direct = images_root / image_name
    if direct.exists():
        return direct

    # 2) Try common nested path under images/.
    nested = images_root / "images" / image_name
    if nested.exists():
        return nested

    # 3) Fallback to basename index (handles unknown subfolder layout).
    candidates = image_index.get(Path(image_name).name.lower(), [])
    if candidates:
        return candidates[0]

    return None


def _collect_pairs_for_camera(gtsfm_data: GtsfmData, camera_idx: int) -> list[MeasurementPair]:
    camera = gtsfm_data.get_camera(camera_idx)
    if camera is None:
        return []

    pairs: list[MeasurementPair] = []
    for track_idx, measurement_idx in gtsfm_data.get_measurements_for_camera(camera_idx):
        track = gtsfm_data.get_track(track_idx)
        _, uv_measured = track.measurement(measurement_idx)
        uv_reproj, success = camera.projectSafe(track.point3())
        if not success:
            continue

        measured = np.asarray(uv_measured, dtype=float)
        reprojected = np.asarray(uv_reproj, dtype=float)
        error = float(np.linalg.norm(measured - reprojected))
        pairs.append(MeasurementPair(track_idx=track_idx, measured=measured, reprojected=reprojected, error=error))
    return pairs


def _track_color(track_idx: int) -> tuple[int, int, int]:
    """Deterministic RGB color for each track id."""
    hue = (track_idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _select_adjacent_cameras(
    candidate_camera_indices: list[int],
    per_camera_mean_err: list[tuple[int, float]],
    max_images: int,
) -> list[int]:
    """Pick a consecutive camera window centered around the worst-error camera."""
    if not candidate_camera_indices:
        return []

    ordered = sorted(candidate_camera_indices)
    if len(ordered) <= max_images:
        return ordered

    anchor = per_camera_mean_err[0][0] if per_camera_mean_err else ordered[len(ordered) // 2]
    anchor_pos = ordered.index(anchor) if anchor in ordered else len(ordered) // 2
    half = max_images // 2
    start = max(0, anchor_pos - half)
    end = start + max_images
    if end > len(ordered):
        end = len(ordered)
        start = end - max_images
    return ordered[start:end]


def _draw_pairs_on_image(
    image_rgb: np.ndarray,
    pairs: list[MeasurementPair],
    scale_u: float,
    scale_v: float,
    line_width: int,
    dot_radius: int,
) -> np.ndarray:
    image_rgb = image_rgb.astype(np.uint8)

    if cv2 is not None:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        for pair in pairs:
            x_meas = int(round(pair.measured[0] * scale_u))
            y_meas = int(round(pair.measured[1] * scale_v))
            x_rep = int(round(pair.reprojected[0] * scale_u))
            y_rep = int(round(pair.reprojected[1] * scale_v))
            r, g, b = _track_color(pair.track_idx)
            color_bgr = (b, g, r)

            # Use unique track color. Reprojected point is slightly larger.
            cv2.line(
                image_bgr,
                (x_rep, y_rep),
                (x_meas, y_meas),
                color_bgr,
                thickness=line_width,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                image_bgr,
                (x_meas, y_meas),
                dot_radius,
                (160, 160, 160),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                image_bgr,
                (x_rep, y_rep),
                dot_radius + 1,
                color_bgr,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    pil_image = PILImage.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    for pair in pairs:
        x_meas = float(pair.measured[0] * scale_u)
        y_meas = float(pair.measured[1] * scale_v)
        x_rep = float(pair.reprojected[0] * scale_u)
        y_rep = float(pair.reprojected[1] * scale_v)
        color = _track_color(pair.track_idx)
        draw.line([(x_rep, y_rep), (x_meas, y_meas)], fill=color, width=line_width)
        draw.ellipse(
            [(x_meas - dot_radius, y_meas - dot_radius), (x_meas + dot_radius, y_meas + dot_radius)],
            fill=(160, 160, 160),
        )
        draw.ellipse(
            [(x_rep - (dot_radius + 1), y_rep - (dot_radius + 1)), (x_rep + (dot_radius + 1), y_rep + (dot_radius + 1))],
            fill=color,
        )
    return np.asarray(pil_image)


def _compose_correspondence_canvas(
    items: list[tuple[int, np.ndarray, dict[int, tuple[float, float]]]],
    max_correspondences_per_pair: int,
    rng: np.random.Generator,
    point_radius: int,
    line_width: int,
) -> tuple[np.ndarray, int]:
    """Create a single stitched image with adjacent-image correspondence lines."""
    if not items:
        return np.zeros((1, 1, 3), dtype=np.uint8), 0

    target_h = int(min(img.shape[0] for _, img, _ in items))
    resized_images: list[np.ndarray] = []
    track_to_xy_per_image: list[dict[int, tuple[float, float]]] = []
    widths: list[int] = []

    for _, image_rgb, track_xy in items:
        h, w = image_rgb.shape[:2]
        if h != target_h:
            new_w = max(1, int(round(w * (target_h / h))))
            if cv2 is not None:
                resized = cv2.resize(image_rgb, (new_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                resized = np.asarray(PILImage.fromarray(image_rgb).resize((new_w, target_h), PILImage.Resampling.BILINEAR))
        else:
            resized = image_rgb
            new_w = w

        sx = new_w / w if w > 0 else 1.0
        sy = target_h / h if h > 0 else 1.0

        track_map: dict[int, tuple[float, float]] = {}
        for tid, (x0, y0) in track_xy.items():
            x = float(x0 * sx)
            y = float(y0 * sy)
            track_map[tid] = (x, y)

        resized_images.append(resized.astype(np.uint8))
        track_to_xy_per_image.append(track_map)
        widths.append(new_w)

    gap = 24
    canvas_w = int(sum(widths) + gap * (len(widths) - 1))
    canvas_h = target_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    x_offsets: list[int] = []
    x_cursor = 0
    for img in resized_images:
        h, w = img.shape[:2]
        canvas[:h, x_cursor : x_cursor + w] = img
        x_offsets.append(x_cursor)
        x_cursor += w + gap

    correspondences_drawn = 0

    if cv2 is not None:
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        for i in range(len(track_to_xy_per_image) - 1):
            a_map = track_to_xy_per_image[i]
            b_map = track_to_xy_per_image[i + 1]
            common = list(set(a_map.keys()).intersection(b_map.keys()))
            if not common:
                continue
            if len(common) > max_correspondences_per_pair:
                sampled = rng.choice(len(common), size=max_correspondences_per_pair, replace=False)
                common = [common[int(k)] for k in sampled]

            for tid in common:
                ax, ay = a_map[tid]
                bx, by = b_map[tid]
                p1 = (int(round(ax + x_offsets[i])), int(round(ay)))
                p2 = (int(round(bx + x_offsets[i + 1])), int(round(by)))
                r, g, b = _track_color(tid)
                color_bgr = (b, g, r)
                cv2.line(canvas_bgr, p1, p2, color_bgr, thickness=line_width, lineType=cv2.LINE_AA)
                cv2.circle(canvas_bgr, p1, point_radius + 1, color_bgr, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(canvas_bgr, p2, point_radius + 1, color_bgr, thickness=-1, lineType=cv2.LINE_AA)
                correspondences_drawn += 1

        return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB), correspondences_drawn

    pil_canvas = PILImage.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    for i in range(len(track_to_xy_per_image) - 1):
        a_map = track_to_xy_per_image[i]
        b_map = track_to_xy_per_image[i + 1]
        common = list(set(a_map.keys()).intersection(b_map.keys()))
        if not common:
            continue
        if len(common) > max_correspondences_per_pair:
            sampled = rng.choice(len(common), size=max_correspondences_per_pair, replace=False)
            common = [common[int(k)] for k in sampled]

        for tid in common:
            ax, ay = a_map[tid]
            bx, by = b_map[tid]
            x1, y1 = float(ax + x_offsets[i]), float(ay)
            x2, y2 = float(bx + x_offsets[i + 1]), float(by)
            color = _track_color(tid)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
            draw.ellipse(
                [(x1 - (point_radius + 1), y1 - (point_radius + 1)), (x1 + (point_radius + 1), y1 + (point_radius + 1))],
                fill=color,
            )
            draw.ellipse(
                [(x2 - (point_radius + 1), y2 - (point_radius + 1)), (x2 + (point_radius + 1), y2 + (point_radius + 1))],
                fill=color,
            )
            correspondences_drawn += 1

    return np.asarray(pil_canvas), correspondences_drawn


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _in_image_bounds(uv: np.ndarray, width: int, height: int) -> bool:
    x, y = float(uv[0]), float(uv[1])
    return 0.0 <= x < float(width) and 0.0 <= y < float(height)


def _compute_oob_sanity(
    per_camera: dict[int, list[MeasurementPair]],
    gtsfm_data: GtsfmData,
    recon_hw_by_image: dict[int, tuple[int, int]],
) -> dict[str, float]:
    measured_oob = 0
    reprojected_oob = 0
    total = 0
    per_image_measured_oob_max = 0.0
    per_image_reprojected_oob_max = 0.0

    for camera_idx, pairs in per_camera.items():
        if not pairs:
            continue
        info = gtsfm_data.get_image_info(camera_idx)
        if camera_idx in recon_hw_by_image:
            h, w = recon_hw_by_image[camera_idx]
        elif info.shape is not None:
            h, w = info.shape
        else:
            continue
        if w <= 0 or h <= 0:
            continue

        img_measured_oob = 0
        img_reprojected_oob = 0
        for pair in pairs:
            total += 1
            if not _in_image_bounds(pair.measured, w, h):
                measured_oob += 1
                img_measured_oob += 1
            if not _in_image_bounds(pair.reprojected, w, h):
                reprojected_oob += 1
                img_reprojected_oob += 1

        img_total = len(pairs)
        if img_total > 0:
            per_image_measured_oob_max = max(per_image_measured_oob_max, img_measured_oob / img_total)
            per_image_reprojected_oob_max = max(per_image_reprojected_oob_max, img_reprojected_oob / img_total)

    measured_oob_rate = (measured_oob / total) if total > 0 else float("nan")
    reprojected_oob_rate = (reprojected_oob / total) if total > 0 else float("nan")

    return {
        "measured_oob_count": float(measured_oob),
        "reprojected_oob_count": float(reprojected_oob),
        "measured_oob_rate": float(measured_oob_rate),
        "reprojected_oob_rate": float(reprojected_oob_rate),
        "per_image_measured_oob_rate_max": float(per_image_measured_oob_max),
        "per_image_reprojected_oob_rate_max": float(per_image_reprojected_oob_max),
    }


def _compute_adjacent_pair_displacement_sanity(
    selected_cameras: list[int],
    per_camera: dict[int, list[MeasurementPair]],
) -> dict[str, float]:
    if len(selected_cameras) < 2:
        return {
            "adjacent_pair_count": 0.0,
            "adjacent_pair_min_common_tracks": float("nan"),
            "adjacent_pair_median_common_tracks": float("nan"),
            "adjacent_pair_disp_median_px": float("nan"),
            "adjacent_pair_disp_p90_px": float("nan"),
            "adjacent_pair_disp_max_px": float("nan"),
        }

    pair_common_counts: list[int] = []
    all_displacements: list[float] = []
    for cam_a, cam_b in zip(selected_cameras[:-1], selected_cameras[1:]):
        a_map = {p.track_idx: p.measured for p in per_camera.get(cam_a, [])}
        b_map = {p.track_idx: p.measured for p in per_camera.get(cam_b, [])}
        common_ids = set(a_map.keys()).intersection(b_map.keys())
        pair_common_counts.append(len(common_ids))
        if not common_ids:
            continue
        for tid in common_ids:
            all_displacements.append(float(np.linalg.norm(a_map[tid] - b_map[tid])))

    if pair_common_counts:
        pair_common_arr = np.asarray(pair_common_counts, dtype=float)
        min_common = float(np.min(pair_common_arr))
        median_common = float(np.median(pair_common_arr))
    else:
        min_common = float("nan")
        median_common = float("nan")

    if all_displacements:
        disp_arr = np.asarray(all_displacements, dtype=float)
        disp_median = float(np.median(disp_arr))
        disp_p90 = float(np.percentile(disp_arr, 90))
        disp_max = float(np.max(disp_arr))
    else:
        disp_median = float("nan")
        disp_p90 = float("nan")
        disp_max = float("nan")

    return {
        "adjacent_pair_count": float(max(0, len(selected_cameras) - 1)),
        "adjacent_pair_min_common_tracks": min_common,
        "adjacent_pair_median_common_tracks": median_common,
        "adjacent_pair_disp_median_px": disp_median,
        "adjacent_pair_disp_p90_px": disp_p90,
        "adjacent_pair_disp_max_px": disp_max,
    }


def _save_histogram(
    errors: list[float],
    output_path: Path,
    title: str,
    bins: int,
    clip_px: float,
) -> bool:
    """Save histogram plot if matplotlib is available."""
    if plt is None or not errors:
        return False

    arr = np.asarray(errors, dtype=float)
    shown = np.clip(arr, 0.0, clip_px)
    mean_v = float(np.mean(arr))
    median_v = float(np.median(arr))
    p90_v = float(np.percentile(arr, 90))
    p95_v = float(np.percentile(arr, 95))
    std_v = float(np.std(arr))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    n_v = int(arr.size)

    fig = plt.figure(figsize=(8, 4.5), dpi=140)
    ax = fig.add_subplot(111)
    ax.hist(shown, bins=bins, color="#2f80ed", edgecolor="#1b4f9c", alpha=0.9)
    ax.axvline(mean_v, color="#0b1f3b", linestyle="-", linewidth=1.6, label=f"mean={mean_v:.3f}")
    ax.axvline(median_v, color="#f2994a", linestyle="--", linewidth=1.5, label=f"median={median_v:.3f}")
    ax.axvline(p90_v, color="#27ae60", linestyle="--", linewidth=1.3, label=f"p90={p90_v:.3f}")
    ax.axvline(p95_v, color="#eb5757", linestyle="--", linewidth=1.3, label=f"p95={p95_v:.3f}")
    ax.set_title(title)
    ax.set_xlabel(f"Reprojection error (px), clipped to {clip_px:g}")
    ax.set_ylabel("Measurements")
    ax.grid(True, alpha=0.25)
    stats_text = (
        f"n={n_v}\n"
        f"min={min_v:.3f}px\n"
        f"max={max_v:.3f}px\n"
        f"std={std_v:.3f}px\n"
        f"mean={mean_v:.3f}px\n"
        f"median={median_v:.3f}px\n"
        f"p90={p90_v:.3f}px\n"
        f"p95={p95_v:.3f}px"
    )
    ax.text(
        0.985,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#7f8c8d"},
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return True


def _process_cluster(
    model_dir: Path,
    recon_root: Path,
    images_root: Path,
    output_root: Path,
    max_images: int,
    max_tracks_per_image: int,
    line_width: int,
    dot_radius: int,
    max_correspondences_per_pair: int,
    corr_low_residual_only: bool,
    corr_residual_thresh_px: float | None,
    hist_bins: int,
    hist_clip_px: float,
    rng: np.random.Generator,
    image_index: dict[str, list[Path]],
) -> tuple[dict[str, object], list[float]]:
    cluster_rel = model_dir.relative_to(recon_root)
    cluster_output_dir = output_root / cluster_rel
    cluster_output_dir.mkdir(parents=True, exist_ok=True)

    gtsfm_data = GtsfmData.read_colmap(str(model_dir))
    recon_hw_by_image = _parse_colmap_recon_sizes(model_dir)
    camera_indices = sorted(gtsfm_data.get_valid_camera_indices())

    all_errors: list[float] = []
    per_camera: dict[int, list[MeasurementPair]] = {}
    per_camera_mean_err: list[tuple[int, float]] = []

    for camera_idx in camera_indices:
        pairs = _collect_pairs_for_camera(gtsfm_data, camera_idx)
        if not pairs:
            continue
        per_camera[camera_idx] = pairs
        camera_err = float(np.mean([pair.error for pair in pairs]))
        per_camera_mean_err.append((camera_idx, camera_err))
        all_errors.extend(pair.error for pair in pairs)

    stats = _stats(all_errors)
    corr_residual_thresh = float(stats["p95"]) if corr_residual_thresh_px is None else float(corr_residual_thresh_px)
    summary: dict[str, object] = {
        "cluster": str(cluster_rel),
        "num_cameras": len(camera_indices),
        "num_tracks": int(gtsfm_data.number_tracks()),
        "num_measurements": int(stats["count"]),
        "mean_reproj_error_px": stats["mean"],
        "median_reproj_error_px": stats["median"],
        "p90_reproj_error_px": stats["p90"],
        "p95_reproj_error_px": stats["p95"],
        "max_reproj_error_px": stats["max"],
        "viz_images": [],
        "histogram_plot": "",
        "correspondence_plot": "",
        "correspondence_lines": 0,
        "measured_correspondence_plot": "",
        "measured_correspondence_lines": 0,
        "corr_residual_thresh_px": corr_residual_thresh if corr_low_residual_only else float("nan"),
        "missing_images": 0,
        "saved_overlay_count": 0,
        "measured_oob_count": 0,
        "reprojected_oob_count": 0,
        "measured_oob_rate": float("nan"),
        "reprojected_oob_rate": float("nan"),
        "per_image_measured_oob_rate_max": float("nan"),
        "per_image_reprojected_oob_rate_max": float("nan"),
        "adjacent_pair_count": 0,
        "adjacent_pair_min_common_tracks": float("nan"),
        "adjacent_pair_median_common_tracks": float("nan"),
        "adjacent_pair_disp_median_px": float("nan"),
        "adjacent_pair_disp_p90_px": float("nan"),
        "adjacent_pair_disp_max_px": float("nan"),
    }

    # Choose adjacent images centered around the worst reprojection-error camera.
    per_camera_mean_err.sort(key=lambda x: x[1], reverse=True)
    selected_cameras = _select_adjacent_cameras(list(per_camera.keys()), per_camera_mean_err, max_images=max_images)
    summary.update(_compute_oob_sanity(per_camera=per_camera, gtsfm_data=gtsfm_data, recon_hw_by_image=recon_hw_by_image))
    summary.update(_compute_adjacent_pair_displacement_sanity(selected_cameras=selected_cameras, per_camera=per_camera))
    composed_items: list[tuple[int, np.ndarray, dict[int, tuple[float, float]]]] = []
    composed_items_measured: list[tuple[int, np.ndarray, dict[int, tuple[float, float]]]] = []

    for camera_idx in selected_cameras:
        info = gtsfm_data.get_image_info(camera_idx)
        if info.name is None:
            continue

        image_path = _get_image_path(info.name, images_root, image_index)
        if image_path is None:
            summary["missing_images"] = int(summary["missing_images"]) + 1
            continue

        if cv2 is not None:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.asarray(PILImage.open(image_path).convert("RGB"))

        loaded_h, loaded_w = image_rgb.shape[:2]
        recon_h: int
        recon_w: int
        if camera_idx in recon_hw_by_image:
            recon_h, recon_w = recon_hw_by_image[camera_idx]
        elif info.shape is not None:
            recon_h, recon_w = info.shape
        else:
            recon_h, recon_w = loaded_h, loaded_w

        # Draw only on reconstruction-resolution images.
        if recon_w > 0 and recon_h > 0 and (loaded_w != recon_w or loaded_h != recon_h):
            if cv2 is not None:
                image_rgb = cv2.resize(image_rgb, (recon_w, recon_h), interpolation=cv2.INTER_AREA)
            else:
                image_rgb = np.asarray(
                    PILImage.fromarray(image_rgb).resize((recon_w, recon_h), PILImage.Resampling.BILINEAR)
                )
        scale_u = 1.0
        scale_v = 1.0

        pairs = per_camera[camera_idx]
        if len(pairs) > max_tracks_per_image:
            sampled_indices = rng.choice(len(pairs), size=max_tracks_per_image, replace=False)
            sampled_pairs = [pairs[int(i)] for i in sampled_indices]
        else:
            sampled_pairs = pairs

        overlay_rgb = _draw_pairs_on_image(
            image_rgb=image_rgb,
            pairs=sampled_pairs,
            scale_u=scale_u,
            scale_v=scale_v,
            line_width=line_width,
            dot_radius=dot_radius,
        )

        stem = Path(info.name).stem
        out_name = f"{camera_idx:06d}_{stem}_tracks.jpg"
        out_path = cluster_output_dir / out_name
        if cv2 is not None:
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        else:
            PILImage.fromarray(overlay_rgb).save(out_path, quality=95)
        summary["viz_images"].append(str(out_path.relative_to(output_root)))
        summary["saved_overlay_count"] = int(summary["saved_overlay_count"]) + 1
        track_xy: dict[int, tuple[float, float]] = {}
        track_xy_measured: dict[int, tuple[float, float]] = {}
        for p in pairs:
            if corr_low_residual_only and p.error > corr_residual_thresh:
                continue
            track_xy[p.track_idx] = (float(p.reprojected[0]), float(p.reprojected[1]))
            track_xy_measured[p.track_idx] = (float(p.measured[0]), float(p.measured[1]))
        composed_items.append((camera_idx, image_rgb, track_xy))
        composed_items_measured.append((camera_idx, image_rgb, track_xy_measured))

    if len(composed_items) >= 2:
        composed_items.sort(key=lambda x: x[0])
        corr_img, corr_count = _compose_correspondence_canvas(
            items=composed_items,
            max_correspondences_per_pair=max_correspondences_per_pair,
            rng=rng,
            point_radius=dot_radius,
            line_width=max(1, line_width),
        )
        corr_path = cluster_output_dir / "adjacent_reprojected_correspondences.jpg"
        if cv2 is not None:
            cv2.imwrite(str(corr_path), cv2.cvtColor(corr_img, cv2.COLOR_RGB2BGR))
        else:
            PILImage.fromarray(corr_img).save(corr_path, quality=95)
        summary["correspondence_plot"] = str(corr_path.relative_to(output_root))
        summary["correspondence_lines"] = int(corr_count)

    if len(composed_items_measured) >= 2:
        composed_items_measured.sort(key=lambda x: x[0])
        corr_img_m, corr_count_m = _compose_correspondence_canvas(
            items=composed_items_measured,
            max_correspondences_per_pair=max_correspondences_per_pair,
            rng=rng,
            point_radius=dot_radius,
            line_width=max(1, line_width),
        )
        corr_path_m = cluster_output_dir / "adjacent_measured_correspondences.jpg"
        if cv2 is not None:
            cv2.imwrite(str(corr_path_m), cv2.cvtColor(corr_img_m, cv2.COLOR_RGB2BGR))
        else:
            PILImage.fromarray(corr_img_m).save(corr_path_m, quality=95)
        summary["measured_correspondence_plot"] = str(corr_path_m.relative_to(output_root))
        summary["measured_correspondence_lines"] = int(corr_count_m)

    hist_path = cluster_output_dir / "reprojection_hist.png"
    if _save_histogram(
        errors=all_errors,
        output_path=hist_path,
        title=f"Cluster {cluster_rel} reprojection error",
        bins=hist_bins,
        clip_px=hist_clip_px,
    ):
        summary["histogram_plot"] = str(hist_path.relative_to(output_root))

    return summary, all_errors


def _write_reports(output_root: Path, summaries: list[dict[str, object]], global_errors: list[float], bins: int, clip_px: float) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "cluster_reprojection_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    csv_path = output_root / "cluster_reprojection_summary.csv"
    fieldnames = [
        "cluster",
        "num_cameras",
        "num_tracks",
        "num_measurements",
        "mean_reproj_error_px",
        "median_reproj_error_px",
        "p90_reproj_error_px",
        "p95_reproj_error_px",
        "max_reproj_error_px",
        "viz_images",
        "histogram_plot",
        "correspondence_plot",
        "correspondence_lines",
        "measured_correspondence_plot",
        "measured_correspondence_lines",
        "corr_residual_thresh_px",
        "missing_images",
        "saved_overlay_count",
        "measured_oob_count",
        "reprojected_oob_count",
        "measured_oob_rate",
        "reprojected_oob_rate",
        "per_image_measured_oob_rate_max",
        "per_image_reprojected_oob_rate_max",
        "adjacent_pair_count",
        "adjacent_pair_min_common_tracks",
        "adjacent_pair_median_common_tracks",
        "adjacent_pair_disp_median_px",
        "adjacent_pair_disp_p90_px",
        "adjacent_pair_disp_max_px",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            row["viz_images"] = ";".join(summary.get("viz_images", []))
            writer.writerow(row)

    _save_histogram(
        errors=global_errors,
        output_path=output_root / "global_reprojection_hist.png",
        title="Global reprojection error across clusters",
        bins=bins,
        clip_px=clip_px,
    )


def main() -> None:
    args = _parse_args()

    recon_root = Path(args.recon_root)
    images_root = Path(args.images_root)
    output_root = Path(args.output_root) if args.output_root else recon_root / "check_tracks"
    rng = np.random.default_rng(args.random_seed)
    image_index = _build_image_index(images_root)
    print(f"Indexed {sum(len(v) for v in image_index.values())} images under {images_root}")

    model_dirs = _find_model_dirs(recon_root, args.model_name)
    if not model_dirs:
        raise ValueError(f"No COLMAP model directories named '{args.model_name}' found under: {recon_root}")

    summaries: list[dict[str, object]] = []
    global_errors: list[float] = []
    for model_dir in model_dirs:
        try:
            summary, cluster_errors = _process_cluster(
                model_dir=model_dir,
                recon_root=recon_root,
                images_root=images_root,
                output_root=output_root,
                max_images=args.max_images,
                max_tracks_per_image=args.max_tracks_per_image,
                line_width=args.line_width,
                dot_radius=args.dot_radius,
                max_correspondences_per_pair=args.max_correspondences_per_pair,
                corr_low_residual_only=args.corr_low_residual_only,
                corr_residual_thresh_px=args.corr_residual_thresh_px,
                hist_bins=args.hist_bins,
                hist_clip_px=args.hist_clip_px,
                rng=rng,
                image_index=image_index,
            )
            summaries.append(summary)
            global_errors.extend(cluster_errors)
            print(
                f"[OK] {summary['cluster']}: "
                f"tracks={summary['num_tracks']}, "
                f"measurements={summary['num_measurements']}, "
                f"mean={summary['mean_reproj_error_px']:.3f}px, "
                f"p95={summary['p95_reproj_error_px']:.3f}px, "
                f"overlays={summary['saved_overlay_count']}/{args.max_images}, "
                f"corr_lines={summary['correspondence_lines']}, "
                f"missing_images={summary['missing_images']}, "
                f"oob(m/r)={100.0 * float(summary['measured_oob_rate']):.2f}%/"
                f"{100.0 * float(summary['reprojected_oob_rate']):.2f}%, "
                f"adj_min_common={int(float(summary['adjacent_pair_min_common_tracks'])) if not np.isnan(float(summary['adjacent_pair_min_common_tracks'])) else -1}"
            )
        except Exception as exc:
            print(f"[FAIL] {model_dir}: {exc}")

    summaries.sort(key=lambda s: float(s["mean_reproj_error_px"]), reverse=True)
    _write_reports(output_root, summaries, global_errors=global_errors, bins=args.hist_bins, clip_px=args.hist_clip_px)

    print(f"\nSaved reports to: {output_root}")
    print(f"  - {output_root / 'cluster_reprojection_summary.json'}")
    print(f"  - {output_root / 'cluster_reprojection_summary.csv'}")
    if plt is not None:
        print(f"  - {output_root / 'global_reprojection_hist.png'}")
    else:
        print("  - matplotlib not available; histogram plots were skipped")


if __name__ == "__main__":
    main()
