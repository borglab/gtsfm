"""Visualize reprojection errors for tracks stored in COLMAP text outputs.

This script reconstructs a GtsfmData object from COLMAP text files, builds a
single loader from a Hydra config, and overlays reprojection error vectors on
each image. Each measurement draws a line between the reprojected point and the
measured 2D keypoint, with an optional dot for the track.

The script searches `--result_root` recursively for folders containing COLMAP
`cameras.txt`, `images.txt`, and `points3D.txt`, then writes visualizations to
`<result_root>/tracks_viz/...` mirroring the COLMAP folder structure.
"""

from __future__ import annotations

import argparse
import colorsys
import os
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import cv2
import hydra
import numpy as np
from hydra.utils import instantiate
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.loader_base import LoaderBase

logger = logger_utils.get_logger()


def _build_loader(
    loader_config: str,
    dataset_dir: str,
    images_dir: str | None,
    max_resolution: int | None,
) -> LoaderBase:
    """Instantiate a loader using a Hydra config."""
    overrides: List[str] = [f"dataset_dir={dataset_dir}"]
    if images_dir is not None:
        overrides.append(f"images_dir={images_dir}")
    if max_resolution is not None:
        overrides.append(f"max_resolution={max_resolution}")

    config_dir = Path(__file__).resolve().parents[1] / "configs" / "loader"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name=loader_config, overrides=overrides)
    return instantiate(cfg)


def _collect_reprojection_pairs(
    gtsfm_data: GtsfmData,
    camera_idx: int,
    allowed_track_indices: Set[int],
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Collect (track_idx, measured, reprojected) for a given camera index."""
    camera = gtsfm_data.get_camera(camera_idx)
    if camera is None:
        return []

    pairs: List[Tuple[int, np.ndarray, np.ndarray]] = []
    measurements = gtsfm_data.get_measurements_for_camera(camera_idx)
    for track_idx, measurement_idx in measurements:
        if track_idx not in allowed_track_indices:
            continue
        track = gtsfm_data.get_track(track_idx)
        image_idx, uv_measured = track.measurement(measurement_idx)
        assert image_idx == camera_idx, "Measurement image index does not match camera index"
        uv_reproj, success = camera.projectSafe(track.point3())
        if not success:
            continue
        pairs.append((track_idx, np.array(uv_measured, dtype=float), np.array(uv_reproj, dtype=float)))
    return pairs


def _track_color(track_idx: int) -> Tuple[int, int, int]:
    """Assign a consistent, distinguishable RGB color per track index."""
    hue = (track_idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _draw_reprojection_overlay(
    image_array: np.ndarray,
    pairs: Iterable[Tuple[int, np.ndarray, np.ndarray]],
    *,
    line_color: Tuple[int, int, int],
    dot_radius: int,
    line_width: int,
    draw_measured: bool,
    measured_color: Tuple[int, int, int],
    scale_u: float,
    scale_v: float,
    dot_on_measured: bool,
    line_only: bool,
) -> PILImageType:
    """Draw reprojection overlays on an image using OpenCV."""
    image_rgb = image_array.astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    line_color_bgr = (line_color[2], line_color[1], line_color[0])
    measured_color_bgr = (measured_color[2], measured_color[1], measured_color[0])

    for track_idx, uv_measured, uv_reproj in pairs:
        x_meas = float(uv_measured[0]) * scale_u
        y_meas = float(uv_measured[1]) * scale_v
        x_rep = float(uv_reproj[0]) * scale_u
        y_rep = float(uv_reproj[1]) * scale_v
        reproj_color = _track_color(track_idx)
        dot_x, dot_y = (x_meas, y_meas) if dot_on_measured else (x_rep, y_rep)

        pt_rep = (int(round(x_rep)), int(round(y_rep)))
        pt_meas = (int(round(x_meas)), int(round(y_meas)))
        pt_dot = (int(round(dot_x)), int(round(dot_y)))

        cv2.line(image_bgr, pt_rep, pt_meas, line_color_bgr, thickness=line_width, lineType=cv2.LINE_AA)
        if not line_only:
            reproj_color_bgr = (reproj_color[2], reproj_color[1], reproj_color[0])
            cv2.circle(image_bgr, pt_dot, dot_radius, reproj_color_bgr, thickness=-1, lineType=cv2.LINE_AA)
            if draw_measured:
                cv2.circle(
                    image_bgr,
                    pt_meas,
                    dot_radius,
                    measured_color_bgr,
                    thickness=max(1, line_width),
                    lineType=cv2.LINE_AA,
                )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(image_rgb)


def _resolve_output_name(gtsfm_data: GtsfmData, loader: LoaderBase, image_idx: int) -> str:
    """Resolve output filename based on COLMAP image names, with loader fallback."""
    info = gtsfm_data.get_image_info(image_idx)
    if info.name:
        return Path(info.name).name
    filenames = loader.image_filenames()
    if 0 <= image_idx < len(filenames):
        name = Path(filenames[image_idx]).name
        if name:
            return name
    return f"image_{image_idx:06d}.png"


def _build_loader_name_maps(loader: LoaderBase) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Build lookup maps from loader filenames to loader indices."""
    filenames = loader.image_filenames()
    full_map: dict[str, int] = {}
    base_map: dict[str, list[int]] = {}
    for idx, name in enumerate(filenames):
        full_map[name] = idx
        base = Path(name).name
        base_map.setdefault(base, []).append(idx)
    return full_map, base_map


def _resolve_loader_index(
    gtsfm_data: GtsfmData, image_idx: int, full_map: dict[str, int], base_map: dict[str, list[int]]
) -> int | None:
    """Resolve loader index for a COLMAP image index based on filename."""
    info = gtsfm_data.get_image_info(image_idx)
    if info.name:
        if info.name in full_map:
            return full_map[info.name]
        base = Path(info.name).name
        if base in base_map:
            if len(base_map[base]) > 1:
                logger.warning("Multiple loader matches for %s; using first.", base)
            return base_map[base][0]
    return None


def _has_colmap_text_files(directory: str) -> bool:
    """Check whether a directory contains COLMAP text outputs."""
    required = {"cameras.txt", "images.txt", "points3D.txt"}
    try:
        entries = set(os.listdir(directory))
    except FileNotFoundError:
        return False
    return required.issubset(entries)


def _find_colmap_dirs(root_dir: str) -> List[str]:
    """Recursively find all subdirectories containing COLMAP text files."""
    matches: List[str] = []
    for dirpath, _, _ in os.walk(root_dir):
        if _has_colmap_text_files(dirpath):
            matches.append(dirpath)
    return matches


def _visualize_tracks_for_dir(args: argparse.Namespace, colmap_dir: str, output_dir: str, loader: LoaderBase) -> None:
    """Visualize reprojection errors for one COLMAP directory."""
    logger.info("Loading reconstruction from %s", colmap_dir)
    try:
        gtsfm_data = GtsfmData.read_colmap(colmap_dir)
    except Exception as exc:
        logger.exception("Skipping %s due to error: %s", colmap_dir, exc)
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_map, base_map = _build_loader_name_maps(loader)

    camera_indices = sorted(gtsfm_data.get_valid_camera_indices())
    if args.max_images is not None:
        camera_indices = camera_indices[: args.max_images]

    num_tracks = gtsfm_data.number_tracks()
    if args.max_pairs is not None and num_tracks > args.max_pairs:
        rng = np.random.default_rng(args.random_seed)
        sampled = rng.choice(num_tracks, size=args.max_pairs, replace=False)
        allowed_track_indices = set(int(idx) for idx in sampled)
    else:
        allowed_track_indices = set(range(num_tracks))

    for camera_idx in camera_indices:
        loader_idx = _resolve_loader_index(gtsfm_data, camera_idx, full_map, base_map)
        if loader_idx is None:
            logger.warning("Skipping camera %d with no loader match", camera_idx)
            continue

        pairs = _collect_reprojection_pairs(gtsfm_data, camera_idx, allowed_track_indices)
        if not pairs:
            logger.info("No valid measurements for image %d", camera_idx)
            continue

        image = loader.get_image(loader_idx)
        resized_h, resized_w = image.height, image.width
        info = gtsfm_data.get_image_info(camera_idx)
        if info.shape is not None:
            orig_h, orig_w = info.shape
        else:
            orig_h, orig_w = resized_h, resized_w
        scale_u = resized_w / orig_w if orig_w > 0 else 1.0
        scale_v = resized_h / orig_h if orig_h > 0 else 1.0

        overlay = _draw_reprojection_overlay(
            image.value_array,
            pairs,
            line_color=tuple(args.line_color),
            dot_radius=args.dot_radius,
            line_width=args.line_width,
            draw_measured=args.draw_measured,
            measured_color=tuple(args.measured_color),
            scale_u=scale_u,
            scale_v=scale_v,
            dot_on_measured=args.dot_on_measured,
            line_only=args.line_only,
        )

        output_name = _resolve_output_name(gtsfm_data, loader, camera_idx)
        output_file = output_path / output_name
        overlay.save(output_file)
        logger.info("Saved %s", output_file)


def visualize_tracks(args: argparse.Namespace) -> None:
    """Visualize reprojection errors across all COLMAP directories under result_root."""
    colmap_dirs = _find_colmap_dirs(args.result_root)
    if not colmap_dirs:
        logger.warning("No COLMAP text directories found under %s", args.result_root)
        return

    logger.info("Instantiating loader config=%s", args.loader_config)
    loader = _build_loader(
        loader_config=args.loader_config,
        dataset_dir=args.dataset_dir,
        images_dir=args.images_dir,
        max_resolution=args.max_resolution,
    )

    viz_root = Path(args.result_root) / "tracks_viz"
    for colmap_dir in colmap_dirs:
        rel_path = Path(colmap_dir).relative_to(args.result_root)
        output_dir = viz_root / rel_path
        _visualize_tracks_for_dir(args, colmap_dir, str(output_dir), loader)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay reprojection error vectors on images for COLMAP reconstructions."
    )
    parser.add_argument(
        "--result_root",
        type=str,
        required=True,
        help="Root directory to recursively search for COLMAP text outputs.",
    )
    parser.add_argument(
        "--loader_config",
        type=str,
        default="colmap",
        help="Loader config name from gtsfm/configs/loader (e.g., colmap, tanks_and_temples).",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset root for the loader (passed as loader.dataset_dir).",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Optional images directory (passed as loader.images_dir).",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=None,
        help="Optional max resolution override for loader.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit the number of images to visualize.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Limit the number of tracks drawn across all images (randomly sampled).",
    )
    parser.add_argument("--dot_radius", type=int, default=2, help="Radius for reprojection dot.")
    parser.add_argument("--line_width", type=int, default=1, help="Line width for reprojection error.")
    parser.add_argument(
        "--line_color",
        type=int,
        nargs=3,
        default=(255, 0, 0),
        help="RGB color for reprojection error lines.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for sampling tracks when max_pairs is set.",
    )
    parser.add_argument(
        "--draw_measured",
        action="store_true",
        help="Draw an outline circle at the measured 2D point.",
    )
    parser.add_argument(
        "--dot_on_measured",
        action="store_true",
        help="Draw the colored dot on the measured point instead of the reprojection.",
    )
    parser.add_argument(
        "--line_only",
        action="store_true",
        help="Draw only the line; use a small line-colored dot at the line head.",
    )
    parser.add_argument(
        "--measured_color",
        type=int,
        nargs=3,
        default=(0, 255, 0),
        help="RGB color for measured point outlines.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    visualize_tracks(_parse_args())
