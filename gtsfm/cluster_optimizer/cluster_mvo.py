"""Multi-view optimization cluster implementation used in the SceneOptimizer pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import numpy as np
from dask.delayed import Delayed, delayed
from dask.distributed import Future, worker_client
from gtsam import Pose3, Similarity3  # type: ignore

import gtsfm.common.types as gtsfm_types
import gtsfm.two_view_estimator as two_view_estimator
import gtsfm.utils.ellipsoid as ellipsoid_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.viz as viz_utils
from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterComputationGraph, ClusterContext, ClusterOptimizerBase
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.types import create_camera
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.two_view_result import TwoViewResult
from gtsfm.products.visibility_graph import AnnotatedGraph, VisibilityGraph
from gtsfm.two_view_estimator import TwoViewEstimator, create_two_view_estimator_futures
from gtsfm.ui.gtsfm_process import UiMetadata
from gtsfm.utils import transform

logger = logger_utils.get_logger()


@dataclass(frozen=True)
class FrontendGraphs:
    """Delayed front-end outputs consumed by downstream stages."""

    keypoints: Delayed
    padded_keypoints: Delayed
    two_view_results: Delayed
    runtime_metrics: Delayed


class ClusterMVO(ClusterOptimizerBase):
    """Handles optimization and I/O for a single leaf cluster using the traditional MVO pipeline."""

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
        drop_child_if_merging_fail: bool = True,
        drop_camera_with_no_track: bool = True,
        drop_outlier_after_camera_merging: bool = True,
        plot_reprojection_histograms: bool = True,
        run_bundle_adjustment_on_parent: bool = True,
    ) -> None:
        # correspondence_generator is MVO-specific; do not pass it to base class.
        super().__init__(
            pose_angular_error_thresh=pose_angular_error_thresh,
            output_worker=output_worker,
            drop_child_if_merging_fail=drop_child_if_merging_fail,
            drop_camera_with_no_track=drop_camera_with_no_track,
            drop_outlier_after_camera_merging=drop_outlier_after_camera_merging,
            plot_reprojection_histograms=plot_reprojection_histograms,
            run_bundle_adjustment_on_parent=run_bundle_adjustment_on_parent,
        )
        # assign MVO-only correspondence generator on this instance
        self.correspondence_generator = correspondence_generator
        self.two_view_estimator = two_view_estimator
        self.multiview_optimizer = multiview_optimizer
        self.dense_multiview_optimizer = dense_multiview_optimizer
        self.gaussian_splatting_optimizer = gaussian_splatting_optimizer
        self._save_two_view_viz = save_two_view_viz

        self._save_gtsfm_data = save_gtsfm_data
        self._save_3d_viz = save_3d_viz

    @property
    def correspondence_generator(self) -> Optional[Any]:
        """Return the registered correspondence generator, if any."""
        return self._correspondence_generator

    @correspondence_generator.setter
    def correspondence_generator(self, value: Optional[Any]) -> None:
        self._correspondence_generator = value

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
        return "ClusterMVO(\n  " + ",\n  ".join(components) + "\n)"

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""
        return UiMetadata(
            display_name="Multiview Optimizer",
            input_products=("Key Images",),
            output_products=("Bundle Adjustment Result",),
            parent_plate="Cluster Optimizer",
        )

    @staticmethod
    def _run_correspondence_generator(
        correspondence_generator: CorrespondenceGeneratorBase,
        visibility_graph: VisibilityGraph,
        image_future_keys: list[str],
    ) -> tuple[list[Keypoints], AnnotatedGraph[np.ndarray], float]:
        """Execute correspondence generation inside a worker task."""

        logger.info("🔵 Running correspondence generation for %d pairs.", len(visibility_graph))

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
        logger.info("🔵 Running two-view estimation for %d pairs.", len(putative_corr_idxs_dict))

        with worker_client() as nested_client:
            start_time = time.time()
            two_view_result_futures = create_two_view_estimator_futures(
                client=nested_client,
                two_view_estimator=two_view_estimator,
                keypoints_list=keypoints_list,
                putative_corr_idxs_dict=putative_corr_idxs_dict,
                relative_pose_priors=relative_pose_priors,
                gt_scene_mesh=gt_scene_mesh,
                one_view_data_dict=one_view_data_dict,
            )
            gathered_tve_futures = nested_client.gather(two_view_result_futures)
            duration_sec = time.time() - start_time

        all_two_view_results = cast(AnnotatedGraph[TwoViewResult], gathered_tve_futures)
        valid_two_view_results = {edge: result for edge, result in all_two_view_results.items() if result.valid()}

        if len(valid_two_view_results) == 0:
            logger.warning("🔵 ClusterMVO: Skipping cluster as it has no valid two-view results.")

        return valid_two_view_results, duration_sec

    @staticmethod
    def _save_two_view_visualizations(
        images_by_index: dict[int, Image],
        two_view_results: AnnotatedGraph[TwoViewResult],
        keypoints_list: list[Keypoints],
        output_dir: Path,
    ) -> None:
        """Persist two-view correspondence visualizations for all valid edges."""
        logger.info("🔵 ClusterMVO: Saving two-view correspondences visualizations to %s.", output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        for (i1, i2), output in two_view_results.items():
            if i1 not in images_by_index or i2 not in images_by_index:
                logger.warning("Images for indices %d or %d missing; skipping visualization.", i1, i2)
                continue
            image_i1 = images_by_index[i1]
            image_i2 = images_by_index[i2]
            viz_utils.save_twoview_correspondences_viz(
                image_i1,
                image_i2,
                keypoints_list[i1],
                keypoints_list[i2],
                output.v_corr_idxs,
                two_view_report=output.post_isp_report,
                file_path=str(output_dir / f"{i1}_{i2}__{image_i1.file_name}_{image_i2.file_name}.jpg"),
            )

    def _build_frontend_graphs(self, context: ClusterContext) -> FrontendGraphs:
        """Create delayed nodes for the full front-end pipeline."""

        visibility_edges = list(context.visibility_graph)
        image_future_keys = [context.image_future_map[idx].key for idx in range(context.num_images)]

        keypoints_graph, putative_corr_idxs_graph, correspondence_duration_graph = delayed(
            ClusterMVO._run_correspondence_generator, nout=3
        )(
            self.correspondence_generator,
            visibility_edges,
            image_future_keys,
        )

        padded_keypoints_graph = delayed(_pad_keypoints_list)(keypoints_graph, context.num_images)

        relative_pose_priors = context.loader.get_relative_pose_priors(context.visibility_graph) or {}
        gt_scene_mesh = context.loader.get_gt_scene_trimesh()

        two_view_results_graph, two_view_duration_graph = delayed(ClusterMVO._run_two_view_estimation, nout=2)(
            self.two_view_estimator,
            padded_keypoints_graph,
            putative_corr_idxs_graph,
            relative_pose_priors,
            gt_scene_mesh,
            context.one_view_data_dict,
        )

        runtime_metrics_graph = delayed(ClusterMVO._build_frontend_runtime_metrics)(
            correspondence_duration_graph,
            two_view_duration_graph,
        )

        return FrontendGraphs(
            keypoints=keypoints_graph,
            padded_keypoints=padded_keypoints_graph,
            two_view_results=two_view_results_graph,
            runtime_metrics=runtime_metrics_graph,
        )

    @staticmethod
    def _collect_post_isp_reports(
        two_view_results: AnnotatedGraph[TwoViewResult],
    ) -> AnnotatedGraph[TwoViewEstimationReport]:
        """Collect post-ISP reports for metrics aggregation."""

        return {edge: result.post_isp_report for edge, result in two_view_results.items() if result.post_isp_report}

    @staticmethod
    def _annotate_scene_with_images(scene: GtsfmData, images_by_index: Mapping[int, Image]) -> GtsfmData:
        """Populate image metadata and colors using the scene helper."""
        return scene.clone_with_image_data(images_by_index)

    @staticmethod
    def _build_frontend_runtime_metrics(
        correspondence_duration_sec: float,
        two_view_duration_sec: float,
    ) -> GtsfmMetricsGroup:
        """Capture simple runtime metrics for the front-end."""

        return GtsfmMetricsGroup(
            "frontend_runtime_metrics",
            [
                GtsfmMetric("total_correspondence_generation_duration_sec", correspondence_duration_sec),
                GtsfmMetric("total_two_view_estimation_duration_sec", two_view_duration_sec),
            ],
        )

    @staticmethod
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

        # Downstream retrieval diagnostics are handled centrally after cluster execution.

    def create_computation_graph(self, context: ClusterContext) -> ClusterComputationGraph | None:
        """Create Dask graphs for multi-view optimization and downstream products for a single cluster.

        The cluster optimizer now owns the full front-end execution for the provided `visibility_graph`
        (correspondence generation and two-view estimation) before invoking the multi-view optimizer.

        Returns:
            - List of Delayed I/O tasks to be computed
            - List of Delayed metrics to be computed
        """
        frontend_graphs: FrontendGraphs = self._build_frontend_graphs(context=context)

        # Get images for all cluster indices as a delayed computation. within this cluster,
        d_cluster_images = context.get_delayed_image_map()

        # Note: the MultiviewOptimizer returns BA input and BA output aligned to GT via Sim(3).
        (
            ba_input_graph,
            ba_output_graph,
            view_graph_two_view_reports,
            optimizer_metrics_graph,
        ) = self.multiview_optimizer.create_computation_graph(
            keypoints_graph=frontend_graphs.padded_keypoints,  # type: ignore[arg-type]
            two_view_results_graph=frontend_graphs.two_view_results,  # type: ignore[arg-type]
            one_view_data_dict=context.one_view_data_dict,
            image_future_map=context.image_future_map,
            output_root=context.output_paths.plots,
        )

        delayed_io_tasks: list[Delayed] = []
        metrics_graph_list: list[Delayed] = [frontend_graphs.runtime_metrics]  # type: ignore

        if optimizer_metrics_graph is not None:
            metrics_graph_list.extend(optimizer_metrics_graph)

        def enqueue_frontend_report(report_graph: Delayed, tag: str) -> None:
            with self._output_annotation():
                delayed_io_tasks.append(
                    delayed(self.save_full_frontend_metrics)(
                        report_graph,
                        context.one_view_data_dict,
                        filename=f"two_view_report_{tag}.json",
                        metrics_path=context.output_paths.metrics,
                        plot_base_path=context.output_paths.plots,
                    )
                )

        post_isp_reports_graph = delayed(ClusterMVO._collect_post_isp_reports)(frontend_graphs.two_view_results)
        enqueue_frontend_report(post_isp_reports_graph, two_view_estimator.POST_ISP_REPORT_TAG)

        enqueue_frontend_report(view_graph_two_view_reports, two_view_estimator.VIEWGRAPH_REPORT_TAG)

        if self._save_two_view_viz:
            with self._output_annotation():
                delayed_io_tasks.append(
                    delayed(ClusterMVO._save_two_view_visualizations)(
                        d_cluster_images,
                        frontend_graphs.two_view_results,
                        frontend_graphs.padded_keypoints,
                        context.output_paths.plots,
                    )
                )

        # Persist all front-end metrics and their summaries.
        metrics_graph_list.append(
            delayed(two_view_estimator.aggregate_frontend_metrics)(
                post_isp_reports_graph,
                self._pose_angular_error_thresh,
                metric_group_name=f"verifier_summary_{two_view_estimator.POST_ISP_REPORT_TAG}",
            )
        )
        metrics_graph_list.append(
            delayed(two_view_estimator.aggregate_frontend_metrics)(
                view_graph_two_view_reports,
                self._pose_angular_error_thresh,
                metric_group_name=f"verifier_summary_{two_view_estimator.VIEWGRAPH_REPORT_TAG}",
            )
        )

        # Modify BA input, BA output, and GT poses to have point clouds and frustums aligned with x,y,z axes.
        gt_wTi_list = [context.one_view_data_dict[idx].pose_gt for idx in range(context.num_images)]
        ba_input_graph, ba_output_graph, aligned_gt_wTi_list = delayed(align_estimated_gtsfm_data, nout=3)(
            ba_input_graph, ba_output_graph, gt_wTi_list
        )

        # Create I/O tasks.
        ba_output_graph = delayed(ClusterMVO._annotate_scene_with_images)(ba_output_graph, d_cluster_images)
        cameras_gt = [context.one_view_data_dict[idx].camera_gt for idx in range(context.num_images)]
        # Align GT cameras using the same Sim(3)/axis alignment applied to BA outputs so they live in the
        # exported frame and stay consistent with the aligned tracks.
        aligned_cameras_gt = delayed(_align_cameras_to_poses)(cameras_gt, aligned_gt_wTi_list)
        with self._output_annotation():
            if self._save_gtsfm_data:
                delayed_io_tasks.append(
                    delayed(save_gtsfm_data)(
                        ba_input_graph,
                        ba_output_graph,
                        results_path=context.output_paths.results,
                        cameras_gt=aligned_cameras_gt,
                    )
                )
                if self._save_3d_viz:
                    delayed_io_tasks.extend(
                        save_matplotlib_visualizations(
                            aligned_ba_input_graph=ba_input_graph,
                            aligned_ba_output_graph=ba_output_graph,
                            gt_pose_graph=aligned_gt_wTi_list,  # type: ignore[arg-type]
                            plot_ba_input_path=context.output_paths.plots,
                            plot_results_path=context.output_paths.plots,
                        )
                    )

        # Create dense reconstruction tasks.
        if self.dense_multiview_optimizer is not None:
            (
                dense_points_graph,
                dense_point_colors_graph,
                densify_metrics_graph,
                downsampling_metrics_graph,
            ) = self.dense_multiview_optimizer.create_computation_graph(d_cluster_images, ba_output_graph)

            with self._output_annotation():
                delayed_io_tasks.append(
                    delayed(io_utils.save_point_cloud_as_ply)(
                        save_fpath=str(context.output_paths.results / "dense_point_cloud.ply"),
                        points=dense_points_graph,
                        rgb=dense_point_colors_graph,
                    )
                )

            if densify_metrics_graph is not None:
                metrics_graph_list.append(densify_metrics_graph)
            if downsampling_metrics_graph is not None:
                metrics_graph_list.append(downsampling_metrics_graph)

        # Gaussian Splatting optimization and rendering, if asked.
        if self.gaussian_splatting_optimizer is not None:
            # Intentional import here to support mac implementation.
            import gtsfm.splat.rendering as gtsfm_rendering

            splats_graph, cfg_graph = self.gaussian_splatting_optimizer.create_computation_graph(
                d_cluster_images, ba_output_graph
            )

            with self._output_annotation():
                delayed_io_tasks.append(
                    delayed(gtsfm_rendering.save_splats)(save_path=context.output_paths.results, splats=splats_graph)
                )
                delayed_io_tasks.append(
                    delayed(gtsfm_rendering.generate_interpolated_video)(
                        d_cluster_images,
                        ba_output_graph,
                        cfg_graph,
                        splats_graph,
                        str(context.output_paths.results / "interpolated_video.mp4"),
                    )
                )

        return ClusterComputationGraph(
            io_tasks=tuple(delayed_io_tasks),
            metric_tasks=tuple(metrics_graph_list),
            sfm_result=ba_output_graph,
        )


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
    """Align estimated data with ground-truth poses and world axes."""
    # This conversion is OK since the GT poses are expected to be complete.
    gt_poses_dict: dict[int, Pose3] = {i: gt_wTi_list[i] for i in range(len(gt_wTi_list)) if gt_wTi_list[i] is not None}
    w_S_output = ba_output.align_to_poses_via_sim3(gt_poses_dict)
    w_ba_output = ba_output.transform_with_sim3(w_S_output)
    w_S_input = ba_input.align_to_poses_via_sim3(gt_poses_dict)
    w_ba_input = ba_input.transform_with_sim3(w_S_input)

    try:
        aTw = ellipsoid_utils.get_ortho_axis_alignment_transform(w_ba_output)
    except Exception as e:
        aTw = Pose3()
        logger.warning("Could not compute axis alignment transform; skipping. Error: %s", e)
    aSw = Similarity3(R=aTw.rotation(), t=aTw.translation(), s=1.0)
    a_ba_output = w_ba_output.transform_with_sim3(aSw)
    a_ba_input = w_ba_input.transform_with_sim3(aSw)
    a_gt_poses = transform.optional_Pose3s_with_sim3(aSw, gt_wTi_list)

    return a_ba_input, a_ba_output, a_gt_poses


def _align_cameras_to_poses(
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]], aligned_gt_wTi_list: list[Optional[Pose3]]
) -> list[Optional[gtsfm_types.CAMERA_TYPE]]:
    """Transport GT cameras into the aligned frame to match aligned tracks."""
    aligned_cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]] = []
    for aligned_pose, cam in zip(aligned_gt_wTi_list, cameras_gt):
        if aligned_pose is None or cam is None:
            aligned_cameras_gt.append(None)
            continue
        aligned_cameras_gt.append(create_camera(aligned_pose, cam.calibration()))
    return aligned_cameras_gt


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
    # It is okay to use cameras_gt indices, since they are ground truth, and are complete.
    for i, camera in enumerate(cameras_gt):
        if camera is not None:
            gt_gtsfm_data.add_camera(i, camera)
            source_info = ba_output.get_image_info(i)
            if source_info.name is not None or source_info.shape is not None:
                gt_gtsfm_data.set_image_info(i, name=source_info.name, shape=source_info.shape)
    for track in ba_output.get_tracks():
        gt_gtsfm_data.add_track(track)
    return gt_gtsfm_data


def save_gtsfm_data(
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
    ba_input_data.export_as_colmap_text(save_dir=os.path.join(output_dir, "ba_input"))

    # Save the output of Bundle Adjustment.
    ba_output_data.export_as_colmap_text(save_dir=os.path.join(output_dir, "ba_output"))

    # Save the ground truth in the same format, for visualization.
    gt_gtsfm_data = get_gtsfm_data_with_gt_cameras_and_est_tracks(cameras_gt, ba_output_data)
    gt_gtsfm_data.export_as_colmap_text(save_dir=os.path.join(output_dir, "ba_gt"))

    duration_sec = time.time() - start_time
    logger.info("🚀 GtsfmData I/O took %.2f min.", duration_sec / 60.0)
