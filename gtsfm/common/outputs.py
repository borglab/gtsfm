"""Utilities for constructing standard GTSFM output directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class OutputPaths:
    """Container holding filesystem destinations for a (sub-)problem."""

    plot_base: Path
    plot_correspondence: Path
    plot_ba_input: Path
    plot_results: Path
    metrics: Path
    results: Path
    mvs_ply: Path
    gs_path: Path
    interpolated_video: Path


def prepare_output_paths(root: Path, leaf_index: Optional[int]) -> OutputPaths:
    """
    Create directories for the given root (and optional leaf) and return their locations.

    Args:
        root: Base output directory for the run.
        leaf_index: Optional index for the current leaf; if provided, sub-directories are created.

    Returns:
        OutputPaths describing the filesystem locations for plots, metrics, and results.
    """
    leaf_folder = f"leaf_{leaf_index}" if leaf_index is not None else None

    def with_leaf(base: Path) -> Path:
        return base / leaf_folder if leaf_folder else base

    plot_base = with_leaf(root / "plots")
    metrics_path = with_leaf(root / "result_metrics")
    results_path = with_leaf(root / "results")

    plot_correspondence = plot_base / "correspondences"
    plot_ba_input = plot_base / "ba_input"
    plot_results = plot_base / "results"
    mvs_output_dir = results_path / "mvs_output"
    gs_output_dir = results_path / "gs_output"

    for directory in (
        plot_base,
        metrics_path,
        results_path,
        plot_correspondence,
        plot_ba_input,
        plot_results,
        mvs_output_dir,
        gs_output_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return OutputPaths(
        plot_base=plot_base,
        plot_correspondence=plot_correspondence,
        plot_ba_input=plot_ba_input,
        plot_results=plot_results,
        metrics=metrics_path,
        results=results_path,
        mvs_ply=mvs_output_dir / "dense_point_cloud.ply",
        gs_path=gs_output_dir,
        interpolated_video=gs_output_dir / "interpolated_path.mp4",
    )
