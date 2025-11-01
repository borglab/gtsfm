"""Base definitions shared by cluster optimizer variants."""

from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from dask.base import annotate
from dask.delayed import Delayed
from dask.distributed import Client, Future

import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.common.image import Image
from gtsfm.common.outputs import OutputPaths
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.products.visibility_graph import VisibilityGraph
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
    image_futures: tuple[Future, ...]

    @property
    def is_root(self) -> bool:
        return len(self.cluster_path) == 0

    @property
    def results_relative_to_run_root(self) -> Path:
        """Return the cluster results directory relative to the run root."""
        base = self.output_paths.relative_results_path()
        # Ensure we always surface a Path pointing under "results"
        return base

    @property
    def run_root(self) -> Path:
        """Base directory for the entire run."""
        return self.output_paths.run_root()


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
    def get_image_dictionary(image_list: list[Image]) -> dict[int, Image]:
        """Convert a list of images to the MVS input format.

        NOTE: simple helper shared across optimizers to pass images into dense modules.
        """
        return {i: img for i, img in enumerate(image_list)}

    @abstractmethod
    def __repr__(self) -> str:
        """Provide a readable summary of the optimizer configuration."""
        return ""

    @abstractmethod
    def create_computation_graph(
        self,
        context: ClusterContext,
        loader: "LoaderBase",
    ) -> ClusterComputationGraph | None:
        """Create a Dask computation graph to process a cluster.

        Args:
            context: Static metadata for the cluster being scheduled.
            loader: Loader used to fetch image content and auxiliary data.

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
