"""Shared utilities for visualizing track reprojection errors."""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.loader_base import LoaderBase
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


def collect_reprojection_pairs(
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


def track_color(track_idx: int) -> Tuple[int, int, int]:
    """Assign a consistent, distinguishable RGB color per track index."""
    hue = (track_idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_reprojection_overlay(
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
        reproj_color = track_color(track_idx)
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

    image_rgb_out = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(image_rgb_out)


def resolve_output_name(gtsfm_data: GtsfmData, loader: LoaderBase, image_idx: int) -> str:
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


def build_loader_name_maps(loader: LoaderBase) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Build lookup maps from loader filenames to loader indices."""
    filenames = loader.image_filenames()
    full_map: dict[str, int] = {}
    base_map: dict[str, list[int]] = {}
    for idx, name in enumerate(filenames):
        full_map[name] = idx
        base = Path(name).name
        base_map.setdefault(base, []).append(idx)
    return full_map, base_map


def resolve_loader_index(
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


def visualize_reprojection_overlays(
    gtsfm_data: GtsfmData,
    loader: LoaderBase,
    output_dir: str,
    *,
    max_images: int | None = None,
    max_pairs: int | None = None,
    random_seed: int = 0,
    line_color: Tuple[int, int, int] = (255, 0, 0),
    dot_radius: int = 1,
    line_width: int = 2,
    draw_measured: bool = True,
    measured_color: Tuple[int, int, int] = (0, 255, 0),
    dot_on_measured: bool = False,
    line_only: bool = False,
    min_error_px: float = 0.0,
) -> None:
    """Save images with reprojection error overlays for a single reconstruction."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_map, base_map = build_loader_name_maps(loader)

    camera_indices = sorted(gtsfm_data.get_valid_camera_indices())
    if max_images is not None:
        camera_indices = camera_indices[:max_images]

    num_tracks = gtsfm_data.number_tracks()
    if max_pairs is not None and num_tracks > max_pairs:
        rng = np.random.default_rng(random_seed)
        sampled = rng.choice(num_tracks, size=max_pairs, replace=False)
        allowed_track_indices = set(int(idx) for idx in sampled)
    else:
        allowed_track_indices = set(range(num_tracks))

    for camera_idx in camera_indices:
        loader_idx = resolve_loader_index(gtsfm_data, camera_idx, full_map, base_map)
        if loader_idx is None:
            logger.warning("Skipping camera %d with no loader match", camera_idx)
            continue

        pairs = collect_reprojection_pairs(gtsfm_data, camera_idx, allowed_track_indices)
        original_count = len(pairs)
        if min_error_px > 0.0:
            filtered_pairs = []
            for track_idx, uv_measured, uv_reproj in pairs:
                err = float(np.linalg.norm(uv_measured - uv_reproj))
                if err >= min_error_px:
                    filtered_pairs.append((track_idx, uv_measured, uv_reproj))
            pairs = filtered_pairs
            pct = 0.0 if original_count == 0 else (len(pairs) / original_count) * 100.0
            logger.info(
                "Camera %d: visualized %d/%d tracks (%.1f%%) with min_error_px=%.3f",
                camera_idx,
                len(pairs),
                original_count,
                pct,
                min_error_px,
            )
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

        overlay = draw_reprojection_overlay(
            image.value_array,
            pairs,
            line_color=line_color,
            dot_radius=dot_radius,
            line_width=line_width,
            draw_measured=draw_measured,
            measured_color=measured_color,
            scale_u=scale_u,
            scale_v=scale_v,
            dot_on_measured=dot_on_measured,
            line_only=line_only,
        )

        output_name = resolve_output_name(gtsfm_data, loader, camera_idx)
        output_file = output_path / output_name
        overlay.save(output_file)
        logger.info("Saved %s", output_file)
