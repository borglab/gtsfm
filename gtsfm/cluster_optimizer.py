"""Utilities for solving individual view-graph clusters within the SceneOptimizer pipeline."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dask.base import annotate
from dask.delayed import Delayed, delayed
from dask.distributed import Future, worker_client
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
from gtsfm.common.outputs import Outputs
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.evaluation.retrieval_metrics import save_retrieval_two_view_metrics
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.two_view_result import TwoViewResult
from gtsfm.products.visibility_graph import AnnotatedGraph, VisibilityGraph
from gtsfm.two_view_estimator import TwoViewEstimator, run_two_view_estimator_as_futures


@dataclass(frozen=True)
class FrontendGraphs:
    """Delayed front-end outputs consumed by downstream stages."""

    keypoints: Delayed
    padded_keypoints: Delayed
    two_view_results: Delayed
    runtime_task: Optional[Delayed]


# Paths to Save Output in React Folders.
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics"
REACT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "public" / "results"

logger = logger_utils.get_logger()


class ClusterOptimizer:
    """Handles optimization and I/O for a single leaf cluster produced by the graph partitioner."""

    def __init__(
        self,
        correspondence_generator: CorrespondenceGeneratorBase,
        two_view_estimator: TwoViewEstimator,
        multiview_optimizer: MultiViewOptimizer,
        dense_multiview_optimizer: Optional[MVSBase] = None,
        gaussian_splatting_optimizer: Optional[Any] = None,
        save_gtsfm_data: bool = True,
        save_3d_viz: bool = False,
        save_two_view_viz: bool = False,
        pose_angular_error_thresh: float = 3,
        output_worker: Optional[str] = None,
    ) -> None:
        self.correspondence_generator = correspondence_generator
        self.two_view_estimator = two_view_estimator
        self.multiview_optimizer = multiview_optimizer
        self.dense_multiview_optimizer = dense_multiview_optimizer
        self.gaussian_splatting_optimizer = gaussian_splatting_optimizer
        self._save_two_view_viz = save_two_view_viz

        self._save_gtsfm_data = save_gtsfm_data
        self._save_3d_viz = save_3d_viz
        self._pose_angular_error_thresh = pose_angular_error_thresh
        self._output_worker = output_worker

        self.run_dense_optimizer = self.dense_multiview_optimizer is not None
        self.run_gaussian_splatting_optimizer = self.gaussian_splatting_optimizer is not None

    @property
    def pose_angular_error_thresh(self) -> float:
        return self._pose_angular_error_thresh

    def __repr__(self) -> str:
        components = [
            f"correspondence_generator={self.correspondence_generator}",
            f"two_view_estimator={self.two_view_estimator}",
            f"multiview_optimizer={self.multiview_optimizer}",
        ]
        if self.dense_multiview_optimizer is not None:
            components.append(f"dense_multiview_optimizer={self.dense_multiview_optimizer}")
        if self.gaussian_splatting_optimizer is not None:
            components.append(f"gaussian_splatting_optimizer={self.gaussian_splatting_optimizer}")
        return "ClusterOptimizer(\n  " + ",\n  ".join(components) + "\n)"

    @staticmethod
    def _run_correspondence_generator(
        correspondence_generator: CorrespondenceGeneratorBase,
        visibility_graph: VisibilityGraph,
        image_future_keys: list[str],
    ) -> tuple[list[Keypoints], AnnotatedGraph[np.ndarray], float]:
        """Execute correspondence generation inside a worker task."""
        logger.info("🔵 Cluster: running correspondence generation on %d pairs.", len(visibility_graph))

        if len(visibility_graph) == 0:
            return [], {}, 0.0

        with worker_client() as nested_client:
            image_futures = [Future(key=key, client=nested_client) for key in image_future_keys]
            start_time = time.time()
            keypoints_list, putative_corr_idxs_dict = correspondence_generator.generate_correspondences(
                nested_client, image_futures, visibility_graph
            )
            duration_sec = time.time() - start_time

        return keypoints_list, putative_corr_idxs_dict, duration_sec

    @staticmethod
    def _run_two_view_estimation(
        two_view_estimator: TwoViewEstimator,
        keypoints_list: list[Keypoints],
        putative_corr_idxs_dict: AnnotatedGraph[np.ndarray],
        relative_pose_priors: AnnotatedGraph[PosePrior],
        gt_scene_mesh: Optional[Any],
        one_view_data_dict: dict[int, OneViewData],
    ) -> tuple[AnnotatedGraph[TwoViewResult], float]:
        """Execute two-view estimation inside a worker task."""
        logger.info("🔵 Cluster: running two-view estimation on %d pairs.", len(putative_corr_idxs_dict))

        with worker_client() as nested_client:
            start_time = time.time()
            two_view_result_futures = run_two_view_estimator_as_futures(
                client=nested_client,
                two_view_estimator=two_view_estimator,
                keypoints_list=keypoints_list,
                putative_corr_idxs_dict=putative_corr_idxs_dict,
                relative_pose_priors=relative_pose_priors,
                gt_scene_mesh=gt_scene_mesh,
                one_view_data_dict=one_view_data_dict,
            )
            all_two_view_results = nested_client.gather(two_view_result_futures)
            duration_sec = time.time() - start_time

        valid_two_view_results = {
            edge: result for edge, result in all_two_view_results.items() if result.valid()  # type : ignore
        }

        if len(valid_two_view_results) == 0:
            logger.warning("🔵 ClusterOptimizer: Skipping cluster as it has no valid two-view results.")

        return valid_two_view_results, duration_sec

    @staticmethod
    def _save_two_view_visualizations(
        loader: LoaderBase,
        two_view_results: AnnotatedGraph[TwoViewResult],
        keypoints_list: list[Keypoints],
        output_dir: Path,
    ) -> None:
        """Persist two-view correspondence visualizations for all valid edges."""
        logger.info("🔵 ClusterOptimizer: Saving two-view correspondences visualizations to %s.", output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        for (i1, i2), output in two_view_results.items():
            image_i1 = loader.get_image(i1)
            image_i2 = loader.get_image(i2)
            viz_utils.save_twoview_correspondences_viz(
                image_i1,
                image_i2,
                keypoints_list[i1],
                keypoints_list[i2],
                output.v_corr_idxs,
                two_view_report=output.post_isp_report,
                file_path=str(output_dir / f"{i1}_{i2}__{image_i1.file_name}_{image_i2.file_name}.jpg"),
            )

    def _output_annotation(self):
        """Context manager routing heavy I/O to the optional output worker."""

        return annotate(workers=self._output_worker) if self._output_worker else annotate()

    def _build_frontend_graphs(
        self,
        num_images: int,
        one_view_data_dict: dict[int, OneViewData],
        loader: LoaderBase,
        visibility_graph: VisibilityGraph,
        image_futures: list[Future],
        outputs: Outputs,
    ) -> FrontendGraphs:
        """Create delayed nodes for the full front-end pipeline."""

        visibility_edges = list(visibility_graph)
        image_future_keys = [future.key for future in image_futures]

        keypoints_graph, putative_corr_idxs_graph, correspondence_duration_graph = delayed(
            ClusterOptimizer._run_correspondence_generator, nout=3
        )(
            self.correspondence_generator,
            visibility_edges,
            image_future_keys,
        )

        padded_keypoints_graph = delayed(_pad_keypoints_list)(keypoints_graph, num_images)

        relative_pose_priors = loader.get_relative_pose_priors(visibility_graph) or {}
        gt_scene_mesh = loader.get_gt_scene_trimesh()

        two_view_results_graph, two_view_duration_graph = delayed(ClusterOptimizer._run_two_view_estimation, nout=2)(
            self.two_view_estimator,
            padded_keypoints_graph,
            putative_corr_idxs_graph,
            relative_pose_priors,
            gt_scene_mesh,
            one_view_data_dict,
        )

        runtime_task: Optional[Delayed] = None
        if outputs.metrics_sink is not None:
            runtime_task = delayed(ClusterOptimizer._build_frontend_runtime_metrics)(
                correspondence_duration_graph,
                two_view_duration_graph,
                outputs,
            )

        return FrontendGraphs(
            keypoints=keypoints_graph,
            padded_keypoints=padded_keypoints_graph,
            two_view_results=two_view_results_graph,
            runtime_task=runtime_task,
        )

    @staticmethod
    def _collect_post_isp_reports(
        two_view_results: AnnotatedGraph[TwoViewResult],
    ) -> AnnotatedGraph[TwoViewEstimationReport]:
        """Collect post-ISP reports for metrics aggregation."""

        return {edge: result.post_isp_report for edge, result in two_view_results.items() if result.post_isp_report}

    @staticmethod
    def _build_frontend_runtime_metrics(
        correspondence_duration_sec: float,
        two_view_duration_sec: float,
        outputs: Outputs,
    ) -> None:
        """Capture simple runtime metrics for the front-end."""

        sink = outputs.metrics_sink
        if sink is None:
            return

        sink.record(
            GtsfmMetricsGroup(
                "frontend_runtime_metrics",
                [
                    GtsfmMetric("total_correspondence_generation_duration_sec", correspondence_duration_sec),
                    GtsfmMetric("total_two_view_estimation_duration_sec", two_view_duration_sec),
                ],
            )
        )

    def create_computation_graph(
        self,
        num_images: int,
        one_view_data_dict: dict[int, OneViewData],
        output_paths: Outputs,
        loader: LoaderBase,
        output_root: Path,
        visibility_graph: VisibilityGraph,
        image_futures: list[Future],
    ) -> Optional[tuple[Delayed, list[Delayed], list[Delayed]]]:
        """Create Dask graphs for multi-view optimization and downstream products for a single cluster.

        The cluster optimizer now owns the full front-end execution for the provided `visibility_graph`
        (correspondence generation and two-view estimation) before invoking the multi-view optimizer.
        """
        frontend_graphs: FrontendGraphs = self._build_frontend_graphs(
            num_images=num_images,
            one_view_data_dict=one_view_data_dict,
            loader=loader,
            visibility_graph=visibility_graph,
            image_futures=image_futures,
            outputs=output_paths,
        )

        side_effect_tasks: list[Delayed] = []
        if frontend_graphs.runtime_task is not None:
            side_effect_tasks.append(frontend_graphs.runtime_task)  # type: ignore

        # Note: the MultiviewOptimizer returns BA input and BA output aligned to GT via Sim(3).
        image_delayed_map = loader.get_images_as_delayed_map()
        (
            ba_input_graph,
            ba_output_graph,
            view_graph_two_view_reports,
            optimizer_side_effects,
        ) = self.multiview_optimizer.create_computation_graph(
            keypoints_graph=frontend_graphs.padded_keypoints,  # type: ignore[arg-type]
            two_view_results_graph=frontend_graphs.two_view_results,  # type: ignore[arg-type]
            one_view_data_dict=one_view_data_dict,
            image_delayed_map=image_delayed_map,
            outputs=output_paths,
        )

        if optimizer_side_effects:
            side_effect_tasks.extend(optimizer_side_effects)

        delayed_results: list[Delayed] = []

        def enqueue_frontend_report(report_graph: Delayed, tag: str) -> None:
            if output_paths.metrics_sink is None:
                return
            with self._output_annotation():
                delayed_results.append(
                    delayed(save_full_frontend_metrics)(
                        report_graph,
                        one_view_data_dict,
                        filename=f"two_view_report_{tag}.json",
                        metrics_path=output_paths.metrics_dir,
                        plot_base_path=output_paths.plot_base,
                    )
                )

        post_isp_reports_graph = delayed(ClusterOptimizer._collect_post_isp_reports)(frontend_graphs.two_view_results)
        enqueue_frontend_report(post_isp_reports_graph, two_view_estimator.POST_ISP_REPORT_TAG)

        enqueue_frontend_report(view_graph_two_view_reports, two_view_estimator.VIEWGRAPH_REPORT_TAG)

        if self._save_two_view_viz:
            with self._output_annotation():
                delayed_results.append(
                    delayed(ClusterOptimizer._save_two_view_visualizations)(
                        loader,
                        frontend_graphs.two_view_results,
                        frontend_graphs.padded_keypoints,
                        output_paths.plot_correspondence,
                    )
                )

        # Persist all front-end metrics and their summaries.
        if output_paths.metrics_sink is not None:
            side_effect_tasks.append(
                delayed(two_view_estimator.aggregate_frontend_metrics)(
                    view_graph_two_view_reports,
                    self._pose_angular_error_thresh,
                    metric_group_name=f"verifier_summary_{two_view_estimator.VIEWGRAPH_REPORT_TAG}",
                    metrics_sink=output_paths.metrics_sink,
                )
            )

        # Modify BA input, BA output, and GT poses to have point clouds and frustums aligned with x,y,z axes.
        gt_wTi_list = [one_view_data_dict[idx].pose_gt for idx in range(num_images)]
        ba_input_graph, ba_output_graph, aligned_gt_wTi_list = delayed(align_estimated_gtsfm_data, nout=3)(
            ba_input_graph, ba_output_graph, gt_wTi_list
        )

        # Create I/O tasks.
        images = [image_delayed_map[idx] for idx in range(num_images)]
        cameras_gt = [one_view_data_dict[idx].camera_gt for idx in range(num_images)]
        with self._output_annotation():
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
            dense_points_graph, dense_point_colors_graph, densify_task = (
                self.dense_multiview_optimizer.create_computation_graph(
                    img_dict_graph, ba_output_graph, outputs=output_paths
                )
            )

            with self._output_annotation():
                delayed_results.append(
                    delayed(io_utils.save_point_cloud_as_ply)(
                        save_fpath=str(output_paths.mvs_ply),
                        points=dense_points_graph,
                        rgb=dense_point_colors_graph,
                    )
                )

            if densify_task is not None:
                side_effect_tasks.append(densify_task)

        if self.run_gaussian_splatting_optimizer and self.gaussian_splatting_optimizer is not None:
            # Intentional import here to support mac implementation.
            import gtsfm.splat.rendering as gtsfm_rendering

            splats_graph, cfg_graph = self.gaussian_splatting_optimizer.create_computation_graph(
                images, ba_output_graph
            )

            with self._output_annotation():
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

        return ba_output_graph, delayed_results, side_effect_tasks


def get_image_dictionary(image_list: list[Image]) -> dict[int, Image]:
    """Convert a list of images to the MVS input format."""
    return {i: img for i, img in enumerate(image_list)}


def _pad_keypoints_list(keypoints_list: list[Keypoints], target_length: int) -> list[Keypoints]:
    """Pad keypoints list with empty detections so it matches the number of images."""
    if len(keypoints_list) >= target_length:
        return keypoints_list
    padded = list(keypoints_list)
    for _ in range(target_length - len(keypoints_list)):
        padded.append(Keypoints(coordinates=np.zeros((0, 2))))
    return padded


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
