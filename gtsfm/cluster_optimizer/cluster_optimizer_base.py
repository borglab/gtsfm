"""Base definitions shared by cluster optimizer variants."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

from dask.base import annotate
from dask.delayed import Delayed

import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase

# Paths to save output in React folders.
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics"
REACT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "public" / "results"

logger = logger_utils.get_logger()


class ClusterOptimizerBase:
    """Base class for cluster optimizers delivering per-cluster computations."""

    def __init__(
        self,
        correspondence_generator: Optional[CorrespondenceGeneratorBase] = None,
        pose_angular_error_thresh: float = 3.0,
        output_worker: Optional[str] = None,
    ) -> None:
        self._pose_angular_error_thresh = pose_angular_error_thresh
        self._output_worker = output_worker
        self._correspondence_generator = correspondence_generator

    @property
    def pose_angular_error_thresh(self) -> float:
        return self._pose_angular_error_thresh

    @property
    def correspondence_generator(self) -> Optional[Any]:
        """Return the registered correspondence generator, if any."""
        return self._correspondence_generator

    @correspondence_generator.setter
    def correspondence_generator(self, value: Optional[Any]) -> None:
        self._correspondence_generator = value

    def _output_annotation(self):
        """Context manager routing heavy I/O to the optional output worker."""
        return annotate(workers=self._output_worker) if self._output_worker else annotate()

    @abstractmethod
    def __repr__(self) -> str:
        """Provide a readable summary of the optimizer configuration."""

    @abstractmethod
    def create_computation_graph(
        self,
        num_images: int,
        one_view_data_dict,
        output_paths,
        loader,
        output_root: Path,
        visibility_graph,
        image_futures,
    ) -> Optional[Tuple[Delayed, Sequence[Delayed], Sequence[Delayed]]]:
        """Create a Dask computation graph to process a cluster."""


def save_metrics_reports(metrics_group_list: list[GtsfmMetricsGroup], metrics_path: str) -> None:
    """Save metrics to JSON and HTML report for dashboard visualizations."""
    metrics_list = list(metrics_group_list)
    metrics_utils.save_metrics_as_json(metrics_list, metrics_path)
    metrics_utils.save_metrics_as_json(metrics_list, str(REACT_METRICS_PATH))

    metrics_report.generate_metrics_report_html(
        metrics_group_list, os.path.join(metrics_path, "gtsfm_metrics_report.html"), None
    )
