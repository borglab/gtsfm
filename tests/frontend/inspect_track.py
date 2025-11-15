#!/usr/bin/env python3
"""Inspect a single track inside an exported COLMAP-style model. It will print tracks with large error by default.

Example:
    python scripts/inspect_track.py --model-dir results/vggt/ba_input
"""

from __future__ import annotations

import argparse
import math
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

    if track_id is not None:
        track_stats = [_inspect_single_track(data, track_id, offending_only, error_threshold)]
        total_tracks = 1
    else:
        total_tracks = data.number_tracks()
        track_stats = [
            _inspect_single_track(data, idx, offending_only, error_threshold) for idx in range(total_tracks)
        ]

    for offending, reproj_offending, depth_offending in track_stats:
        offending_tracks += int(offending)
        reproj_offenders += int(reproj_offending)
        depth_offenders += int(depth_offending)

    return TrackStats(
        total_tracks=total_tracks,
        offending_tracks=offending_tracks,
        reprojection_offenders=reproj_offenders,
        depth_offenders=depth_offenders,
    )


def _inspect_single_track(
    data: GtsfmData,
    track_id: int,
    offending_only: bool,
    error_threshold: float,
) -> Tuple[bool, bool, bool]:
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
        depth: Optional[float] = None
        reproj_error: Optional[float] = None

        uv_measured_tuple: Tuple[float, float]
        if hasattr(uv_measured, "x"):  # gtsam Point2
            uv_measured_tuple = (float(uv_measured.x()), float(uv_measured.y()))
        else:
            uv_measured_tuple = (float(uv_measured[0]), float(uv_measured[1]))

        if camera is None:
            failure_reason = "camera missing"
        else:
            try:
                reproj = camera.project(point)
                if hasattr(reproj, "x"):
                    uv_reproj = (float(reproj.x()), float(reproj.y()))
                else:
                    uv_reproj = (float(reproj[0]), float(reproj[1]))
                reproj_error = float(math.dist(uv_reproj, uv_measured_tuple))
                cam_frame_point = camera.pose().transformTo(point)
                depth = float(cam_frame_point[2])
            except Exception as exc:  # pragma: no cover - gtsam throws rich runtime errors.
                failure_reason = str(exc)

        summary = MeasurementSummary(
            camera_idx=cam_idx,
            image_name=image_info.name,
            uv_measured=uv_measured_tuple,
            uv_reprojected=uv_reproj,
            depth=depth,
            reproj_error=reproj_error,
            failure_reason=failure_reason,
        )
        summaries.append(summary)

    offending, reproj_offending, depth_offending = _classify_offending_measurements(summaries, error_threshold)

    if offending_only and not offending:
        return offending, reproj_offending, depth_offending

    print(f"Track {track_id}: point=({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
    print(f"  Num measurements: {track.numberMeasurements()}")
    for summary in summaries:
        print("  - " + _describe_measurement(summary))
    return offending, reproj_offending, depth_offending


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
    print(
        "Track summary: "
        f"total={stats.total_tracks}, "
        f"offending={stats.offending_tracks}, "
        f"reprojection_offenders={stats.reprojection_offenders}, "
        f"depth_offenders={stats.depth_offenders}"
    )


if __name__ == "__main__":
    main()
