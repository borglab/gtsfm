"""Base definitions shared by cluster optimizer variants."""

from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Tuple

from dask.base import annotate
from dask.delayed import Delayed, delayed
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

    client: Client
    loader: "LoaderBase"
    num_images: int
    output_paths: OutputPaths
    image_future_map: Dict[int, Future]
    one_view_data_dict: dict[int, "OneViewData"]
    cluster_path: tuple[int, ...]
    label: str
    visibility_graph: VisibilityGraph

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

    @staticmethod
    def resolve_visibility_graph_images(
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

    def get_delayed_image_map(self) -> Delayed:
        """Get images for all cluster indices as a delayed computation. Within this cluster,
        Dask will materialize that dictionary exactly once and share it among those downstream tasks.
        """
        return delayed(self.resolve_visibility_graph_images)(
            self.visibility_graph,
            self.image_future_map,
        )


class ClusterOptimizerBase(GTSFMProcess):
    """Base class for cluster optimizers delivering per-cluster computations."""

    def __init__(
        self,
        pose_angular_error_thresh: float = 3.0,
        drop_child_if_merging_fail: bool = True,
        drop_camera_with_no_track: bool = True,
        drop_outlier_after_camera_merging: bool = True,
        plot_reprojection_histograms: bool = True,
        run_bundle_adjustment_on_parent: bool = True,
        output_worker: None | str = None,
    ) -> None:
        self.drop_child_if_merging_fail = drop_child_if_merging_fail
        self.drop_camera_with_no_track = drop_camera_with_no_track
        self.drop_outlier_after_camera_merging = drop_outlier_after_camera_merging
        self.plot_reprojection_histograms = plot_reprojection_histograms
        self.run_bundle_adjustment_on_parent = run_bundle_adjustment_on_parent
        self._pose_angular_error_thresh = pose_angular_error_thresh
        self._output_worker = output_worker

    @property
    def pose_angular_error_thresh(self) -> float:
        return self._pose_angular_error_thresh

    def _output_annotation(self):
        """Context manager routing heavy I/O to the optional output worker."""
        return annotate(workers=self._output_worker) if self._output_worker else annotate()

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
