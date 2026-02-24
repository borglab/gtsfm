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
import os
from pathlib import Path
from typing import List

import hydra
from hydra.utils import instantiate

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.visualization.track_viz_utils import visualize_reprojection_overlays

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

    visualize_reprojection_overlays(
        gtsfm_data=gtsfm_data,
        loader=loader,
        output_dir=output_dir,
        max_images=args.max_images,
        max_pairs=args.max_pairs,
        random_seed=args.random_seed,
        line_color=tuple(args.line_color),
        dot_radius=args.dot_radius,
        line_width=args.line_width,
        draw_measured=args.draw_measured,
        measured_color=tuple(args.measured_color),
        dot_on_measured=args.dot_on_measured,
        line_only=args.line_only,
    )


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
