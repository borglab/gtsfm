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
            prefix = cluster_path[: depth + 1]
            # Shallow levels keep the classic accumulated label (existing drops/tooling
            # unchanged). Deep levels switch to a short depth+index component: the accumulated
            # label repeats the whole prefix at EVERY level, so total path length grows
            # quadratically with depth — a 63-deep tree (Alamo) blows past PATH_MAX while
            # merely creating output dirs. Short components are sibling-unique (same parent,
            # same depth, distinct last index), keeping the nesting unambiguous.
            component = cluster_label(prefix) if len(prefix) <= 8 else f"C{len(prefix)}_{prefix[-1]}"
            cluster_dir = cluster_dir / component

    # For plotting
    output_paths = OutputPaths(
        results=cluster_dir,
        metrics=cluster_dir / "metrics",
        plots=cluster_dir / "plots",
    )
    output_paths.create_directories()
    return output_paths
