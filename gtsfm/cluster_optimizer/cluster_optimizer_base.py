"""Base definitions shared by cluster optimizer variants."""

from __future__ import annotations

import os
import shutil
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from dask.base import annotate
from dask.delayed import Delayed, delayed
from gtsam import Pose3, Similarity3  # type: ignore

import gtsfm.common.types as gtsfm_types
import gtsfm.evaluation.metrics_report as metrics_report
import gtsfm.two_view_estimator as two_view_estimator
import gtsfm.utils.alignment as alignment_utils
import gtsfm.utils.ellipsoid as ellipsoid_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.viz as viz_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.outputs import OutputPaths
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.evaluation.retrieval_metrics import save_retrieval_two_view_metrics
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.two_view_result import TwoViewResult
from gtsfm.products.visibility_graph import AnnotatedGraph

# Paths to save output in React folders.
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics"
REACT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "public" / "results"

logger = logger_utils.get_logger()


class ClusterOptimizerBase:
    """Base class for cluster optimizers delivering per-cluster computations."""

    def __init__(
        self,
        correspondence_generator: Optional[Any] = None,
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
        return ""

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


# ------------------ Shared helper functions ------------------


def get_image_dictionary(image_list: list[Image]) -> dict[int, Image]:
    """Convert a list of images to the MVS input format.

    NOTE: belongs in base because it's a small generic utility used by densify or other
    postprocessing modules.
    """
    return {i: img for i, img in enumerate(image_list)}


def _pad_keypoints_list(keypoints_list: list[Keypoints], target_length: int) -> list[Keypoints]:
    """Pad keypoints list with empty detections so it matches the number of images.

    NOTE: generic helper for producing consistent BA inputs regardless of front-end.
    """
    if len(keypoints_list) >= target_length:
        return keypoints_list
    padded = list(keypoints_list)
    for _ in range(target_length - len(keypoints_list)):
        padded.append(Keypoints(coordinates=np.zeros((0, 2))))
    return padded


def align_estimated_gtsfm_data(
    ba_input: GtsfmData, ba_output: GtsfmData, gt_wTi_list: list[Optional[Pose3]]
) -> tuple[GtsfmData, GtsfmData, list[Optional[Pose3]]]:
    """Align estimated data with ground-truth poses and world axes.

    NOTE: alignment is common postprocessing for outputs from any optimizer.
    """
    ba_input = alignment_utils.align_gtsfm_data_via_Sim3_to_poses(ba_input, gt_wTi_list)
    ba_output = alignment_utils.align_gtsfm_data_via_Sim3_to_poses(ba_output, gt_wTi_list)

    aTw = ellipsoid_utils.get_ortho_axis_alignment_transform(ba_output)
    aSw = Similarity3(R=aTw.rotation(), t=aTw.translation(), s=1.0)
    ba_input = ba_input.apply_Sim3(aSw)
    ba_output = ba_output.apply_Sim3(aSw)
    gt_wTi_list = [aSw.transformFrom(wTi) if wTi is not None else None for wTi in gt_wTi_list]
    return ba_input, ba_output, gt_wTi_list


def save_matplotlib_visualizations(
    aligned_ba_input_graph: Delayed,
    aligned_ba_output_graph: Delayed,
    gt_pose_graph: list[Optional[Delayed]],
    plot_ba_input_path: Path,
    plot_results_path: Path,
) -> list[Delayed]:
    """Visualize bundle adjustment inputs/outputs and GT poses with Matplotlib.

    NOTE: generic plotting helper used by both MVO and VGGT.
    """
    viz_graph_list = []
    viz_graph_list.append(delayed(viz_utils.save_sfm_data_viz)(aligned_ba_input_graph, plot_ba_input_path))
    viz_graph_list.append(delayed(viz_utils.save_sfm_data_viz)(aligned_ba_output_graph, plot_results_path))
    viz_graph_list.append(
        delayed(viz_utils.save_camera_poses_viz)(
            aligned_ba_input_graph, aligned_ba_output_graph, gt_pose_graph, plot_results_path
        )
    )
    return viz_graph_list


def get_gtsfm_data_with_gt_cameras_and_est_tracks(
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    ba_output: GtsfmData,
) -> GtsfmData:
    """Creates GtsfmData object with GT camera poses and estimated tracks.

    NOTE: utility to export GT cameras alongside estimated tracks for visualization.
    """
    gt_gtsfm_data = GtsfmData(number_images=len(cameras_gt))
    for i, camera in enumerate(cameras_gt):
        if camera is not None:
            gt_gtsfm_data.add_camera(i, camera)
    for track in ba_output.get_tracks():
        gt_gtsfm_data.add_track(track)
    return gt_gtsfm_data


def save_gtsfm_data(
    images: list[Image],
    ba_input_data: GtsfmData,
    ba_output_data: GtsfmData,
    results_path: Path,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
) -> None:
    """Saves the Gtsfm data before and after bundle adjustment.

    NOTE: centralize on-disk export and React duplication here.
    """
    start_time = time.time()
    output_dir = str(results_path)

    # Ensure directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Save input to Bundle Adjustment for debugging.
    ba_input_data.export_as_colmap_text(images=images, save_dir=os.path.join(output_dir, "ba_input"))

    # Save the output of Bundle Adjustment.
    ba_output_data.export_as_colmap_text(images=images, save_dir=os.path.join(output_dir, "ba_output"))

    # Save the ground truth in the same format, for visualization.
    gt_gtsfm_data = get_gtsfm_data_with_gt_cameras_and_est_tracks(cameras_gt, ba_output_data)
    gt_gtsfm_data.export_as_colmap_text(images=images, save_dir=os.path.join(output_dir, "ba_gt"))

    # Delete old version of React results directory and save a duplicate copy.
    shutil.rmtree(REACT_RESULTS_PATH, ignore_errors=True)
    try:
        shutil.copytree(src=results_path, dst=REACT_RESULTS_PATH)
    except Exception:
        logger.warning("Could not copy results to REACT_RESULTS_PATH: %s", REACT_RESULTS_PATH)

    duration_sec = time.time() - start_time
    logger.info("🚀 GtsfmData I/O took %.2f min.", duration_sec / 60.0)


def save_full_frontend_metrics(
    two_view_report_dict: AnnotatedGraph[two_view_estimator.TwoViewEstimationReport],
    one_view_data_dict: dict[int, OneViewData],
    filename: str,
    metrics_path: Path,
    plot_base_path: Path,
) -> None:
    """Converts the TwoViewEstimationReports for all image pairs to a dict and saves it as JSON.

    NOTE: central place for frontend metrics serialization and optional retrieval plotting.
    """
    metrics_list = two_view_estimator.get_two_view_reports_summary(two_view_report_dict, one_view_data_dict)

    io_utils.save_json_file(os.path.join(metrics_path, filename), metrics_list)

    # Save duplicate copy within React folder.
    io_utils.save_json_file(os.path.join(REACT_METRICS_PATH, filename), metrics_list)

    gt_available = any(report.R_error_deg is not None for report in two_view_report_dict.values())

    if "VIEWGRAPH_2VIEW_REPORT" in filename and gt_available:
        save_retrieval_two_view_metrics(metrics_path, plot_base_path)


def save_metrics_reports(metrics_group_list: list[GtsfmMetricsGroup], metrics_path: str) -> None:
    """Save metrics to JSON and HTML report for dashboard visualizations.

    NOTE: central helper for persisting metrics used by different optimizers (MVO, VGGT).
    """
    metrics_utils.save_metrics_as_json(metrics_group_list, metrics_path)
    metrics_utils.save_metrics_as_json(metrics_group_list, str(REACT_METRICS_PATH))

    metrics_report.generate_metrics_report_html(
        metrics_group_list, os.path.join(metrics_path, "gtsfm_metrics_report.html"), None
    )
