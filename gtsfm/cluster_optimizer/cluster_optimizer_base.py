"""Base definitions shared by cluster optimizer variants."""

from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Tuple

from dask.base import annotate
from dask.delayed import Delayed
from dask.distributed import Client, Future, get_client

import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.common.image import Image
from gtsfm.common.outputs import OutputPaths, cluster_label
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.products.visibility_graph import VisibilityGraph, visibility_graph_keys
from gtsfm.ui.gtsfm_process import GTSFMProcess

if TYPE_CHECKING:
    from gtsfm.loader.loader_base import LoaderBase
    from gtsfm.products.one_view_data import OneViewData

# Paths to save output in React folders.
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics"
REACT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "public" / "results"

logger = logger_utils.get_logger()


@dataclass(frozen=True)
class ClusterComputationGraph:
    """Container describing the delayed tasks required for a cluster run."""

    io_tasks: Tuple[Delayed, ...]
    metric_tasks: Tuple[Delayed, ...]
    sfm_result: Delayed | None


@dataclass(frozen=True)
class ClusterContext:
    """Static metadata describing a cluster tree node."""

    visibility_graph: VisibilityGraph
    output_paths: OutputPaths
    cluster_path: tuple[int, ...]
    label: str
    client: Client
    num_images: int
    one_view_data_dict: dict[int, "OneViewData"]
    image_future_map: Dict[int, Future]
    loader: "LoaderBase"

    @property
    def is_root(self) -> bool:
        return len(self.cluster_path) == 0

    @property
    def react_results_subdir(self) -> Path:
        """Subdirectory used when mirroring artifacts into the React workspace."""
        subdir = Path("results")
        if self.cluster_path:
            for depth in range(len(self.cluster_path)):
                subdir /= cluster_label(self.cluster_path[: depth + 1])
        return subdir


class ClusterOptimizerBase(GTSFMProcess):
    """Base class for cluster optimizers delivering per-cluster computations."""

    def __init__(
        self,
        pose_angular_error_thresh: float = 3.0,
        output_worker: None | str = None,
    ) -> None:
        self._pose_angular_error_thresh = pose_angular_error_thresh
        self._output_worker = output_worker

    @property
    def pose_angular_error_thresh(self) -> float:
        return self._pose_angular_error_thresh

    def _output_annotation(self):
        """Context manager routing heavy I/O to the optional output worker."""
        return annotate(workers=self._output_worker) if self._output_worker else annotate()

    @staticmethod
    def resolve_visibility_images(
        visibility_graph: VisibilityGraph,
        image_future_map: Mapping[int, Future],
    ) -> dict[int, Image]:
        """Gather the subset of images referenced by a visibility graph.

        Args:
            visibility_graph: Edges describing which cameras participate in the cluster.
            image_future_map: Mapping from camera index to the loader-provided image future.

        Returns:
            Dictionary of realized `Image` objects keyed by camera index. Missing images are skipped.
        """
        indices = sorted(idx for idx in visibility_graph_keys(visibility_graph) if idx in image_future_map)
        if not indices:
            return {}

        futures = [image_future_map[idx] for idx in indices]
        try:
            images = get_client().gather(futures) if futures else []
        except Exception as exc:
            logger.warning("Failed to gather images for indices %s: %s", indices, exc)
            return {}

        return {idx: img for idx, img in zip(indices, images) if img is not None}

    @abstractmethod
    def __repr__(self) -> str:
        """Provide a readable summary of the optimizer configuration."""
        return ""

    @abstractmethod
    def create_computation_graph(
        self,
        context: ClusterContext,
    ) -> ClusterComputationGraph | None:
        """Create a Dask computation graph to process a cluster.

        Args:
            context: Static metadata for the cluster being scheduled.

        Returns:
            ClusterComputationGraph describing delayed I/O, metrics, and the bundle-adjusted result.
        """


def save_metrics_reports(metrics_group_list: list[GtsfmMetricsGroup], metrics_path: str) -> None:
    """Save metrics to JSON and HTML report for dashboard visualizations.

    NOTE: central helper for persisting metrics used by different optimizers (MVO, VGGT).
    """
    metrics_utils.save_metrics_as_json(metrics_group_list, metrics_path)
    metrics_utils.save_metrics_as_json(metrics_group_list, str(REACT_METRICS_PATH))

    metrics_report.generate_metrics_report_html(
        metrics_group_list, os.path.join(metrics_path, "gtsfm_metrics_report.html"), None
    )
