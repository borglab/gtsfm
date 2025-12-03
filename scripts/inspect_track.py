#!/usr/bin/env python3
"""Inspect a single track inside an exported COLMAP-style model. It will print tracks with large error by default.

Example:
    python scripts/inspect_track.py --model-dir results/vggt/ba_input
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gtsfm.common.gtsfm_data import GtsfmData


@dataclass
class MeasurementSummary:
    """Container for per-measurement diagnostics."""

    camera_idx: int
    image_name: Optional[str]
    uv_measured: Tuple[float, float]
    uv_reprojected: Optional[Tuple[float, float]]
    depth: Optional[float]
    reproj_error: Optional[float]
    failure_reason: Optional[str]


@dataclass
class TrackStats:
    total_tracks: int
    offending_tracks: int
    reprojection_offenders: int
    depth_offenders: int
    mean_reproj_error: Optional[float]
    median_reproj_error: Optional[float]
    min_reproj_error: Optional[float]
    max_reproj_error: Optional[float]


def _format_pair(values: Sequence[float]) -> str:
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


def _describe_measurement(summary: MeasurementSummary) -> str:
    parts = [
        f"cam={summary.camera_idx}",
        f"name={summary.image_name or 'unknown'}",
        f"uv={_format_pair(summary.uv_measured)}",
    ]
    if summary.uv_reprojected is not None:
        parts.append(f"reproj={_format_pair(summary.uv_reprojected)}")
    if summary.depth is not None:
        parts.append(f"depth={summary.depth:.4f}")
    if summary.reproj_error is not None:
        parts.append(f"err={summary.reproj_error:.4f}px")
    if summary.failure_reason is not None:
        parts.append(f"failure='{summary.failure_reason}'")
    return ", ".join(parts)


def _visualize_gtsfm_tracks_on_original_frames(
    square_images: torch.Tensor,
    original_coords: torch.Tensor,
    gtsfm_data: GtsfmData,
    image_indices: list[int],
    output_dir: Path,
) -> None:
    """Restore images to native scale and overlay 2D track measurements from ``GtsfmData``."""

    if gtsfm_data.number_tracks() == 0:
        return

    restored = _restore_images_to_original_scale(square_images, original_coords)

    num_frames = len(image_indices)
    index_lookup = {img_idx: local_idx for local_idx, img_idx in enumerate(image_indices)}
    per_frame_measurements: list[list[tuple[tuple[float, float], tuple[int, int, int]]]] = [
        [] for _ in range(num_frames)
    ]
    track_sequences: list[dict[str, Any]] = []

    # Tracks stored in ``gtsfm_data`` are already expressed in the original image coordinates by vggt.py.
    processed_tracks = 0
    for track_idx in range(gtsfm_data.number_tracks()):
        if processed_tracks >= MAX_TRACKS_TO_DRAW:
            break
        track = gtsfm_data.get_track(track_idx)
        if track is None:
            continue
        measurements: list[tuple[int, float, float]] = []
        for meas_idx in range(track.numberMeasurements()):
            img_idx, uv = track.measurement(meas_idx)
            local_idx = index_lookup.get(img_idx)
            if local_idx is None:
                continue
            if hasattr(uv, "x"):
                u = float(uv.x())
                v = float(uv.y())
            elif isinstance(uv, np.ndarray):
                u = float(uv[0])
                v = float(uv[1])
            else:
                # assume tuple-like
                u = float(uv[0])
                v = float(uv[1])
            measurements.append((local_idx, u, v))

        if len(measurements) == 0:
            continue

        color_bgr = _vibrant_bgr_from_index(track_idx)
        measurements = sorted(measurements, key=lambda m: m[0])
        point3 = track.point3() if hasattr(track, "point3") else None

        track_sequences.append(
            {
                "measurements": measurements,
                "color": color_bgr,
                "point3": point3,
            }
        )
        processed_tracks += 1

        for frame_idx, u, v in measurements:
            per_frame_measurements[frame_idx].append(((u, v), color_bgr))

    if not track_sequences:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    frames_per_row = min(4, max(1, num_frames))
    grid_rows = math.ceil(num_frames / frames_per_row)
    max_h = restored.shape[2]
    max_w = restored.shape[3]
    grid_canvas = np.zeros((grid_rows * max_h, frames_per_row * max_w, 3), dtype=np.uint8)

    restored_np = (restored.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

    for local_idx, frame in enumerate(restored_np):
        row = local_idx // frames_per_row
        col = local_idx % frames_per_row
        y0 = row * max_h
        x0 = col * max_w
        grid_canvas[y0 : y0 + frame.shape[0], x0 : x0 + frame.shape[1]] = frame

    grid_canvas = cv2.cvtColor(grid_canvas, cv2.COLOR_RGB2BGR)

    for track_info in track_sequences:
        measurements = track_info["measurements"]
        color_bgr = track_info["color"]
        last_grid_point = None
        for frame_idx, u, v in measurements:
            row = frame_idx // frames_per_row
            col = frame_idx % frames_per_row
            grid_pt = (int(round(col * max_w + u)), int(round(row * max_h + v)))
            cv2.circle(grid_canvas, grid_pt, radius=5, color=color_bgr, thickness=-1)
            if last_grid_point is not None:
                cv2.line(grid_canvas, last_grid_point, grid_pt, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)
            last_grid_point = grid_pt if len(measurements) > 1 else None

    cv2.imwrite(str(output_dir / "tracks_grid.png"), grid_canvas)

    for local_idx, frame in enumerate(restored_np):
        canvas = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        for point_idx, ((u, v), color_bgr) in enumerate(per_frame_measurements[local_idx]):
            if point_idx >= MAX_POINTS_PER_FRAME:
                break
            pt = (int(round(u)), int(round(v)))
            cv2.circle(canvas, pt, radius=5, color=color_bgr, thickness=-1)
        cv2.imwrite(str(output_dir / f"frame_{local_idx:04d}.png"), canvas)

    for local_idx, frame in enumerate(restored_np):
        img_idx = image_indices[local_idx]
        camera = gtsfm_data.get_camera(img_idx)
        if camera is None:
            continue

        canvas = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]
        for track_info in track_sequences:
            point3 = track_info["point3"]
            if point3 is None:
                continue
            try:
                uv = camera.project(point3)
            except Exception:
                continue

            if hasattr(uv, "x"):
                u = float(uv.x())
                v = float(uv.y())
            elif isinstance(uv, np.ndarray):
                u = float(uv[0])
                v = float(uv[1])
            else:
                u = float(uv[0])
                v = float(uv[1])
            if not (0 <= u < width and 0 <= v < height):
                continue

            color_bgr = track_info["color"]
            pt = (int(round(u)), int(round(v)))
            cv2.drawMarker(
                canvas,
                pt,
                color=color_bgr,
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        cv2.imwrite(str(output_dir / f"reproj_{local_idx:04d}.png"), canvas)


def inspect_track(
    model_dir: Path,
    track_id: Optional[int],
    offending_only: bool,
    error_threshold: float,
) -> TrackStats:
    """Load a COLMAP-format model and print diagnostics for one or more tracks."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist")

    data = GtsfmData.read_colmap(str(model_dir))
    offending_tracks = 0
    reproj_offenders = 0
    depth_offenders = 0
    all_reproj_errors: list[float] = []

    if track_id is not None:
        total_tracks = 1
        track_ids: Sequence[int] = [track_id]
    else:
        total_tracks = data.number_tracks()
        track_ids = range(total_tracks)

    for idx in track_ids:
        offending, reproj_offending, depth_offending, track_errors = _inspect_single_track(
            data, idx, offending_only, error_threshold
        )
        offending_tracks += int(offending)
        reproj_offenders += int(reproj_offending)
        depth_offenders += int(depth_offending)
        all_reproj_errors.extend(track_errors)

    if all_reproj_errors:
        mean_error = float(sum(all_reproj_errors)) / len(all_reproj_errors)
        median_error = float(statistics.median(all_reproj_errors))
        min_error = float(min(all_reproj_errors))
        max_error = float(max(all_reproj_errors))
    else:
        mean_error = median_error = min_error = max_error = None

    return TrackStats(
        total_tracks=total_tracks,
        offending_tracks=offending_tracks,
        reprojection_offenders=reproj_offenders,
        depth_offenders=depth_offenders,
        mean_reproj_error=mean_error,
        median_reproj_error=median_error,
        min_reproj_error=min_error,
        max_reproj_error=max_error,
    )


def _inspect_single_track(
    data: GtsfmData,
    track_id: int,
    offending_only: bool,
    error_threshold: float,
) -> Tuple[bool, bool, bool, list[float]]:
    """Print diagnostics for a single track index."""
    if track_id < 0 or track_id >= data.number_tracks():
        raise ValueError(f"Track id {track_id} out of bounds (0..{data.number_tracks() - 1})")

    track = data.get_track(track_id)
    point = track.point3()

    summaries: list[MeasurementSummary] = []
    for m_idx in range(track.numberMeasurements()):
        cam_idx, uv_measured = track.measurement(m_idx)
        camera = data.get_camera(cam_idx)
        image_info = data.get_image_info(cam_idx)
        failure_reason: Optional[str] = None
        uv_reproj: Optional[Tuple[float, float]] = None
        reproj_error: Optional[float] = None

        cam_frame_point = camera.pose().transformTo(point)
        depth = float(cam_frame_point[2])

        try:
            uv_reproj = camera.project(point)
            reproj_error = float(math.dist(uv_reproj, uv_measured))
        except Exception as exc:  # pragma: no cover - gtsam throws rich runtime errors.
            failure_reason = str(exc)

        summary = MeasurementSummary(
            camera_idx=cam_idx,
            image_name=image_info.name,
            uv_measured=uv_measured,
            uv_reprojected=uv_reproj,
            depth=depth,
            reproj_error=reproj_error,
            failure_reason=failure_reason,
        )
        summaries.append(summary)

    offending, reproj_offending, depth_offending = _classify_offending_measurements(summaries, error_threshold)
    track_errors = [summary.reproj_error for summary in summaries if summary.reproj_error is not None]

    if offending_only and not offending:
        return offending, reproj_offending, depth_offending, track_errors

    print(f"Track {track_id}: point=({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
    print(f"  Num measurements: {track.numberMeasurements()}")
    for summary in summaries:
        print("  - " + _describe_measurement(summary))
    return offending, reproj_offending, depth_offending, track_errors


def _classify_offending_measurements(
    summaries: Sequence[MeasurementSummary], error_threshold: float
) -> Tuple[bool, bool, bool]:
    offending = False
    reproj_offending = False
    depth_offending = False
    for summary in summaries:
        if summary.failure_reason is not None:
            offending = True
        if summary.depth is not None and summary.depth < 0:
            depth_offending = True
            offending = True
        if summary.reproj_error is not None and summary.reproj_error > error_threshold:
            reproj_offending = True
            offending = True
    return offending, reproj_offending, depth_offending


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a track inside a COLMAP-style GTSFM export.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Folder containing cameras.txt/images.txt/points3D.txt (e.g. results/<scene>/ba_input).",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Track index to inspect (matches POINT3D_ID inside points3D.txt). "
        "If omitted, prints diagnostics for every track.",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=5.0,
        help="Reprojection error threshold (pixels) for flagging a measurement as offending.",
    )
    parser.add_argument(
        "--offending-only",
        dest="offending_only",
        action="store_true",
        default=True,
        help="Only display tracks containing negative depth, projection failures, or measurements with "
        "reprojection error above --error-threshold (default).",
    )
    parser.add_argument(
        "--show-all",
        dest="offending_only",
        action="store_false",
        help="Disable filtering and display every track.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = inspect_track(args.model_dir, args.track_id, args.offending_only, args.error_threshold)
    if stats.mean_reproj_error is None:
        print("Reprojection error stats: unavailable (no valid measurements).")
    else:
        print(
            "Reprojection error stats (px): "
            f"mean={stats.mean_reproj_error:.4f}, "
            f"median={stats.median_reproj_error:.4f}, "
            f"min={stats.min_reproj_error:.4f}, "
            f"max={stats.max_reproj_error:.4f}"
        )
    print(
        "Track summary: "
        f"total={stats.total_tracks}, "
        f"offending={stats.offending_tracks}, "
        f"reprojection_offenders={stats.reprojection_offenders}, "
        f"depth_offenders={stats.depth_offenders}"
    )


if __name__ == "__main__":
    main()
