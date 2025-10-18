"""Utilities for solving individual view-graph clusters within the SceneOptimizer pipeline."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

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
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.evaluation.retrieval_metrics import save_retrieval_two_view_metrics
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.two_view_result import TwoViewResult
from gtsfm.products.visibility_graph import AnnotatedGraph

# Paths to Save Output in React Folders.
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics"
REACT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "public" / "results"

logger = logger_utils.get_logger()


class ClusterOptimizer:
    """Handles optimization and I/O for a single leaf cluster produced by the graph partitioner."""

    def __init__(
        self,
        multiview_optimizer: MultiViewOptimizer,
        dense_multiview_optimizer: Optional[MVSBase] = None,
        gaussian_splatting_optimizer: Optional[Any] = None,
        save_gtsfm_data: bool = True,
        save_3d_viz: bool = False,
        pose_angular_error_thresh: float = 3,
        output_worker: Optional[str] = None,
    ) -> None:
        self.multiview_optimizer = multiview_optimizer
        self.dense_multiview_optimizer = dense_multiview_optimizer
        self.gaussian_splatting_optimizer = gaussian_splatting_optimizer
        self._save_gtsfm_data = save_gtsfm_data
        self._save_3d_viz = save_3d_viz
        self._pose_angular_error_thresh = pose_angular_error_thresh
        self._output_worker = output_worker

        self.run_dense_optimizer = self.dense_multiview_optimizer is not None
        self.run_gaussian_splatting_optimizer = self.gaussian_splatting_optimizer is not None

    @property
    def pose_angular_error_thresh(self) -> float:
        return self._pose_angular_error_thresh

    def create_computation_graph(
        self,
        keypoints_list: list[Keypoints],
        two_view_results: AnnotatedGraph[TwoViewResult],
        num_images: int,
        one_view_data_dict: dict[int, OneViewData],
        output_paths: OutputPaths,
        relative_pose_priors: AnnotatedGraph[PosePrior],
        loader: LoaderBase,
        output_root: Path,
    ) -> tuple[Delayed, list[Delayed], list[Delayed]]:
        """Create Dask graphs for multi-view optimization and downstream products for a single cluster."""

        delayed_results: list[Delayed] = []

        image_delayed_map = loader.get_images_as_delayed_map()

        # Note: the MultiviewOptimizer returns BA input and BA output aligned to GT via Sim(3).
        (
            ba_input_graph,
            ba_output_graph,
            view_graph_two_view_reports,
            optimizer_metrics_graph,
        ) = self.multiview_optimizer.create_computation_graph(
            keypoints_list=keypoints_list,
            two_view_results=two_view_results,
            one_view_data_dict=one_view_data_dict,
            image_delayed_map=image_delayed_map,
            relative_pose_priors=relative_pose_priors,
            output_root=output_root,
        )
        if view_graph_two_view_reports is not None:
            two_view_reports_post_viewgraph_estimator = view_graph_two_view_reports

        # Persist all front-end metrics and their summaries.
        metrics_graph_list: list[Delayed] = []
        annotation = annotate(workers=self._output_worker) if self._output_worker else annotate()
        with annotation:
            delayed_results.append(
                delayed(save_full_frontend_metrics)(
                    {ij: r.post_isp_report for ij, r in two_view_results.items()},
                    one_view_data_dict,
                    filename=f"two_view_report_{two_view_estimator.POST_ISP_REPORT_TAG}.json",
                    metrics_path=output_paths.metrics,
                    plot_base_path=output_paths.plot_base,
                )
            )

            delayed_results.append(
                delayed(save_full_frontend_metrics)(
                    two_view_reports_post_viewgraph_estimator,  # type: ignore[arg-type]
                    one_view_data_dict,
                    filename=f"two_view_report_{two_view_estimator.VIEWGRAPH_REPORT_TAG}.json",
                    metrics_path=output_paths.metrics,
                    plot_base_path=output_paths.plot_base,
                )
            )
            metrics_graph_list.append(
                delayed(two_view_estimator.aggregate_frontend_metrics)(
                    two_view_reports_post_viewgraph_estimator,  # type: ignore[arg-type]
                    self._pose_angular_error_thresh,
                    metric_group_name=f"verifier_summary_{two_view_estimator.VIEWGRAPH_REPORT_TAG}",
                )
            )

        if optimizer_metrics_graph is not None:
            metrics_graph_list.extend(optimizer_metrics_graph)

        # Modify BA input, BA output, and GT poses to have point clouds and frustums aligned with x,y,z axes.
        gt_wTi_list = [one_view_data_dict[idx].pose_gt for idx in range(num_images)]
        ba_input_graph, ba_output_graph, aligned_gt_wTi_list = delayed(align_estimated_gtsfm_data, nout=3)(
            ba_input_graph, ba_output_graph, gt_wTi_list
        )

        # Create I/O tasks.
        images = [image_delayed_map[idx] for idx in range(num_images)]
        cameras_gt = [one_view_data_dict[idx].camera_gt for idx in range(num_images)]
        annotation = annotate(workers=self._output_worker) if self._output_worker else annotate()
        with annotation:
            if self._save_gtsfm_data:
                delayed_results.append(
                    delayed(save_gtsfm_data)(
                        images,
                        ba_input_graph,
                        ba_output_graph,
                        results_path=output_paths.results,
                        cameras_gt=cameras_gt,
                    )
                )
                if self._save_3d_viz:
                    delayed_results.extend(
                        save_matplotlib_visualizations(
                            aligned_ba_input_graph=ba_input_graph,
                            aligned_ba_output_graph=ba_output_graph,
                            gt_pose_graph=aligned_gt_wTi_list,  # type: ignore[arg-type]
                            plot_ba_input_path=output_paths.plot_ba_input,
                            plot_results_path=output_paths.plot_results,
                        )
                    )

        # Create dense reconstruction tasks.
        if self.run_dense_optimizer and self.dense_multiview_optimizer is not None:
            img_dict_graph = delayed(get_image_dictionary)(images)
            (
                dense_points_graph,
                dense_point_colors_graph,
                densify_metrics_graph,
                downsampling_metrics_graph,
            ) = self.dense_multiview_optimizer.create_computation_graph(img_dict_graph, ba_output_graph)

            annotation = annotate(workers=self._output_worker) if self._output_worker else annotate()
            with annotation:
                delayed_results.append(
                    delayed(io_utils.save_point_cloud_as_ply)(
                        save_fpath=str(output_paths.mvs_ply),
                        points=dense_points_graph,
                        rgb=dense_point_colors_graph,
                    )
                )

            if densify_metrics_graph is not None:
                metrics_graph_list.append(densify_metrics_graph)
            if downsampling_metrics_graph is not None:
                metrics_graph_list.append(downsampling_metrics_graph)

        if self.run_gaussian_splatting_optimizer and self.gaussian_splatting_optimizer is not None:
            # Intentional import here to support mac implementation.
            import gtsfm.splat.rendering as gtsfm_rendering

            splats_graph, cfg_graph = self.gaussian_splatting_optimizer.create_computation_graph(
                images, ba_output_graph
            )

            annotation = annotate(workers=self._output_worker) if self._output_worker else annotate()
            with annotation:
                delayed_results.append(
                    delayed(gtsfm_rendering.save_splats)(save_path=str(output_paths.gs_path), splats=splats_graph)
                )
                delayed_results.append(
                    delayed(gtsfm_rendering.generate_interpolated_video)(
                        images=images,
                        sfm_result_graph=ba_output_graph,
                        cfg_result_graph=cfg_graph,
                        splats_graph=splats_graph,
                        video_fpath=output_paths.interpolated_video,
                    )
                )

        return ba_output_graph, delayed_results, metrics_graph_list


def get_image_dictionary(image_list: list[Image]) -> dict[int, Image]:
    """Convert a list of images to the MVS input format."""
    return {i: img for i, img in enumerate(image_list)}


def align_estimated_gtsfm_data(
    ba_input: GtsfmData, ba_output: GtsfmData, gt_wTi_list: list[Optional[Pose3]]
) -> tuple[GtsfmData, GtsfmData, list[Optional[Pose3]]]:
    """Align estimated data with ground-truth poses and world axes."""
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
    """Visualize bundle adjustment inputs/outputs and GT poses with Matplotlib."""
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
    """Creates GtsfmData object with GT camera poses and estimated tracks."""
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
    """Saves the Gtsfm data before and after bundle adjustment."""
    start_time = time.time()
    output_dir = str(results_path)

    # Ensure directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Save input to Bundle Adjustment for debugging.
    io_utils.export_model_as_colmap_text(
        gtsfm_data=ba_input_data,
        images=images,
        save_dir=os.path.join(output_dir, "ba_input"),
    )

    # Save the output of Bundle Adjustment.
    io_utils.export_model_as_colmap_text(
        gtsfm_data=ba_output_data,
        images=images,
        save_dir=os.path.join(output_dir, "ba_output"),
    )

    # Save the ground truth in the same format, for visualization.
    gt_gtsfm_data = get_gtsfm_data_with_gt_cameras_and_est_tracks(cameras_gt, ba_output_data)

    io_utils.export_model_as_colmap_text(
        gtsfm_data=gt_gtsfm_data,
        images=images,
        save_dir=os.path.join(output_dir, "ba_output_gt"),
    )

    # Delete old version of React results directory and save a duplicate copy.
    shutil.rmtree(REACT_RESULTS_PATH, ignore_errors=True)
    shutil.copytree(src=results_path, dst=REACT_RESULTS_PATH)

    duration_sec = time.time() - start_time
    logger.info("🚀 GtsfmData I/O took %.2f min.", duration_sec / 60.0)


def save_full_frontend_metrics(
    two_view_report_dict: AnnotatedGraph[two_view_estimator.TwoViewEstimationReport],
    one_view_data_dict: dict[int, OneViewData],
    filename: str,
    metrics_path: Path,
    plot_base_path: Path,
) -> None:
    """Converts the TwoViewEstimationReports for all image pairs to a dict and saves it as JSON."""
    metrics_list = two_view_estimator.get_two_view_reports_summary(two_view_report_dict, one_view_data_dict)

    io_utils.save_json_file(os.path.join(metrics_path, filename), metrics_list)

    # Save duplicate copy within React folder.
    io_utils.save_json_file(os.path.join(REACT_METRICS_PATH, filename), metrics_list)

    gt_available = any(report.R_error_deg is not None for report in two_view_report_dict.values())

    if "VIEWGRAPH_2VIEW_REPORT" in filename and gt_available:
        save_retrieval_two_view_metrics(metrics_path, plot_base_path)


def save_metrics_reports(metrics_group_list: list[GtsfmMetricsGroup], metrics_path: str) -> None:
    """Saves metrics to JSON and HTML report."""
    metrics_utils.save_metrics_as_json(metrics_group_list, metrics_path)
    metrics_utils.save_metrics_as_json(metrics_group_list, str(REACT_METRICS_PATH))

    metrics_report.generate_metrics_report_html(
        metrics_group_list, os.path.join(metrics_path, "gtsfm_metrics_report.html"), None
    )
