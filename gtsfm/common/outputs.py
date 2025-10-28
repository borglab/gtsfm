"""Utilities for constructing standard GTSFM output directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class OutputPaths:
    """Container holding filesystem destinations for a (sub-)problem."""

    results: Path
    metrics: Path
    plots: Path

    def create_directories(self) -> None:
        for directory in (
            self.results,
            self.metrics,
            self.plots,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def prepare_output_paths(root: Path, leaf_index: Optional[int]) -> OutputPaths:
    """
    Create directories for the given root (and optional leaf) and return their locations.

    Args:
        root: Base output directory for the run.
        leaf_index: Optional index for the current leaf; if provided, sub-directories are created.

    Returns:
        OutputPaths describing the filesystem locations for plots, metrics, and results.
    """
    base = root / "results"
    cluster_dir = base / f"leaf_{leaf_index}" if leaf_index else base

    # For plotting
    output_paths = OutputPaths(
        results=cluster_dir,
        metrics=cluster_dir / "metrics",
        plots=cluster_dir / "plots",
    )
    output_paths.create_directories()
    return output_paths
