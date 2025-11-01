"""Utilities for constructing standard GTSFM output directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


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

    def run_root(self) -> Path:
        """Return the base directory for the current run (parent of the results tree)."""
        if self.results.name == "results":
            return self.results.parent
        for parent in self.results.parents:
            if parent.name == "results":
                return parent.parent
        return self.results.parent

    def relative_results_path(self) -> Path:
        """Return path to results directory relative to the run root."""
        root = self.run_root()
        try:
            return self.results.relative_to(root)
        except ValueError:
            return Path(self.results.name)


def cluster_label(path: Sequence[int]) -> str:
    """Return a human-readable label like C12 for the given cluster path."""
    return "C" + "".join(f"_{i}" for i in path) if path else "C"


def prepare_output_paths(root: Path, cluster_path: Optional[Sequence[int]] = None) -> OutputPaths:
    """
    Create directories for the given root (and optional cluster path) and return their locations.

    Args:
        root: Base output directory for the run.
        cluster_path: Optional tuple describing a path in the cluster tree.

    Returns:
        OutputPaths describing the filesystem locations for plots, metrics, and results.
    """
    cluster_dir = root / "results"
    if cluster_path:
        for depth in range(len(cluster_path)):
            cluster_dir = cluster_dir / cluster_label(cluster_path[: depth + 1])

    # For plotting
    output_paths = OutputPaths(
        results=cluster_dir,
        metrics=cluster_dir / "metrics",
        plots=cluster_dir / "plots",
    )
    output_paths.create_directories()
    return output_paths
