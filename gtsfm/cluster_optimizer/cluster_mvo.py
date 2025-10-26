"""Multi-view optimization cluster implementation used in the SceneOptimizer pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dask.delayed import Delayed, delayed
from dask.distributed import Future, worker_client

import gtsfm.two_view_estimator as two_view_estimator
import gtsfm.utils.io as io_utils
import gtsfm.utils.viz as viz_utils
from gtsfm.cluster_optimizer.cluster_optimizer_base import (
    ClusterOptimizerBase,
    _pad_keypoints_list,
    align_estimated_gtsfm_data,
    get_image_dictionary,
    logger,
    save_full_frontend_metrics,
    save_gtsfm_data,
    save_matplotlib_visualizations,
)
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.outputs import OutputPaths
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
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
    ) -> None:
        super().__init__(
            correspondence_generator=correspondence_generator,
            pose_angular_error_thresh=pose_angular_error_thresh,
            output_worker=output_worker,
        )
        self.two_view_estimator = two_view_estimator
        self.multiview_optimizer = multiview_optimizer
        self.dense_multiview_optimizer = dense_multiview_optimizer
        self.gaussian_splatting_optimizer = gaussian_splatting_optimizer
        self._save_two_view_viz = save_two_view_viz

        self._save_gtsfm_data = save_gtsfm_data
        self._save_3d_viz = save_3d_viz

        self.run_dense_optimizer = self.dense_multiview_optimizer is not None
        self.run_gaussian_splatting_optimizer = self.gaussian_splatting_optimizer is not None

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
            logger.warning("🔵 ClusterMVO: Skipping cluster as it has no valid two-view results.")

        return valid_two_view_results, duration_sec

    @staticmethod
    def _save_two_view_visualizations(
        loader: LoaderBase,
        two_view_results: AnnotatedGraph[TwoViewResult],
        keypoints_list: list[Keypoints],
        output_dir: Path,
    ) -> None:
        """Persist two-view correspondence visualizations for all valid edges."""
        logger.info("🔵 ClusterMVO: Saving two-view correspondences visualizations to %s.", output_dir)

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

    def _build_frontend_graphs(
        self,
        num_images: int,
        one_view_data_dict: dict[int, OneViewData],
        loader: LoaderBase,
        visibility_graph: VisibilityGraph,
        image_futures: list[Future],
    ) -> FrontendGraphs:
        """Create delayed nodes for the full front-end pipeline."""

        visibility_edges = list(visibility_graph)
        image_future_keys = [future.key for future in image_futures]

        keypoints_graph, putative_corr_idxs_graph, correspondence_duration_graph = delayed(
            ClusterMVO._run_correspondence_generator, nout=3
        )(
            self.correspondence_generator,
            visibility_edges,
            image_future_keys,
        )

        padded_keypoints_graph = delayed(_pad_keypoints_list)(keypoints_graph, num_images)

        relative_pose_priors = loader.get_relative_pose_priors(visibility_graph) or {}
        gt_scene_mesh = loader.get_gt_scene_trimesh()

        two_view_results_graph, two_view_duration_graph = delayed(ClusterMVO._run_two_view_estimation, nout=2)(
            self.two_view_estimator,
            padded_keypoints_graph,
            putative_corr_idxs_graph,
            relative_pose_priors,
            gt_scene_mesh,
            one_view_data_dict,
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

    def create_computation_graph(
        self,
        num_images: int,
        one_view_data_dict: dict[int, OneViewData],
        output_paths: OutputPaths,
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
        )

        # Note: the MultiviewOptimizer returns BA input and BA output aligned to GT via Sim(3).
        image_delayed_map = loader.get_images_as_delayed_map()
        (
            ba_input_graph,
            ba_output_graph,
            view_graph_two_view_reports,
            optimizer_metrics_graph,
        ) = self.multiview_optimizer.create_computation_graph(
            keypoints_graph=frontend_graphs.padded_keypoints,  # type: ignore[arg-type]
            two_view_results_graph=frontend_graphs.two_view_results,  # type: ignore[arg-type]
            one_view_data_dict=one_view_data_dict,
            image_delayed_map=image_delayed_map,
            output_root=output_root,
        )

        metrics_graph_list: list[Delayed] = [frontend_graphs.runtime_metrics]  # type: ignore
        delayed_results: list[Delayed] = []

        if optimizer_metrics_graph is not None:
            metrics_graph_list.extend(optimizer_metrics_graph)

        def enqueue_frontend_report(report_graph: Delayed, tag: str) -> None:
            with self._output_annotation():
                delayed_results.append(
                    delayed(save_full_frontend_metrics)(
                        report_graph,
                        one_view_data_dict,
                        filename=f"two_view_report_{tag}.json",
                        metrics_path=output_paths.metrics,
                        plot_base_path=output_paths.plot_base,
                    )
                )

        post_isp_reports_graph = delayed(ClusterMVO._collect_post_isp_reports)(frontend_graphs.two_view_results)
        enqueue_frontend_report(post_isp_reports_graph, two_view_estimator.POST_ISP_REPORT_TAG)

        enqueue_frontend_report(view_graph_two_view_reports, two_view_estimator.VIEWGRAPH_REPORT_TAG)

        if self._save_two_view_viz:
            with self._output_annotation():
                delayed_results.append(
                    delayed(ClusterMVO._save_two_view_visualizations)(
                        loader,
                        frontend_graphs.two_view_results,
                        frontend_graphs.padded_keypoints,
                        output_paths.plot_correspondence,
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
            (
                dense_points_graph,
                dense_point_colors_graph,
                densify_metrics_graph,
                downsampling_metrics_graph,
            ) = self.dense_multiview_optimizer.create_computation_graph(img_dict_graph, ba_output_graph)

            with self._output_annotation():
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

        return ba_output_graph, delayed_results, metrics_graph_list
