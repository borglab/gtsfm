"""Base definitions shared by cluster optimizer variants."""

from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Mapping, Tuple

from dask.base import annotate
from dask.delayed import Delayed, delayed
from dask.distributed import Client, Future, get_client

import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.common.image import Image
from gtsfm.common.outputs import OutputPaths
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.products.visibility_graph import VisibilityGraph, visibility_graph_keys
from gtsfm.ui.gtsfm_process import GTSFMProcess

if TYPE_CHECKING:
    from gtsfm.loader.loader_base import LoaderBase
    from gtsfm.products.one_view_data import OneViewData

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
    # Global Fetzer focals (cam idx -> calibration), computed once over the full verified view graph.
    # Used by ClusterVGGTWithFrontend when use_global_view_graph_calibration is set, in place of the
    # per-cluster calibration (which falls back to VGGT focals for cameras with few in-cluster F-edges).
    global_refined_intrinsics: dict | None = None
    # Global frontend products from the verified two-view pass, reused per cluster instead of re-running
    # the per-cluster correspondence generation. v_corr keyed by image-pair; keypoints indexed by image.
    # Used by ClusterVGGTWithFrontend when reuse_global_correspondences is set. See ClusterContext plumbing.
    global_v_corr_idxs_dict: dict | None = None
    global_keypoints: list | None = None
    # Per-cluster slice of the packed (camera, pixel) -> global-track-id index (sorted int64 keys, int32
    # gids). Attached to this cluster's reconstruction as a sidecar so merges can match tracks by GLOBAL
    # identity across clusters that share no cameras. None disables ID-matching (non-verified pipelines).
    measurement_gid_index: tuple | None = None

    @property
    def is_root(self) -> bool:
        return len(self.cluster_path) == 0

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
        output_worker: None | str = None,
    ) -> None:
        self._output_worker = output_worker

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

    metrics_report.generate_metrics_report_html(
        metrics_group_list, os.path.join(metrics_path, "gtsfm_metrics_report.html"), None
    )
