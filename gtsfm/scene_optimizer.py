"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""

import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dask.base import annotate
from dask.delayed import Delayed, delayed
from dask.distributed import Future, performance_report
from gtsam import Pose3, Similarity3  # type: ignore
from trimesh import Trimesh

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
from gtsfm.common.outputs import OutputPaths, prepare_output_paths
from gtsfm.common.pose_prior import PosePrior
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.graph_partitioner.single_partitioner import SinglePartitioner
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.two_view_result import TwoViewResult
from gtsfm.products.visibility_graph import AnnotatedGraph
from gtsfm.retriever.image_pairs_generator import ImagePairsGenerator
from gtsfm.two_view_estimator import (
    POST_ISP_REPORT_TAG,
    VIEWGRAPH_REPORT_TAG,
    TwoViewEstimationReport,
    TwoViewEstimator,
    run_two_view_estimator_as_futures,
)
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator

# Set matplotlib backend to "Agg" (Anti-Grain Geometry) for headless rendering
# This must be called before importing pyplot or any other matplotlib modules
# "Agg" is a non-interactive backend that renders to files without requiring a display
matplotlib.use("Agg")

DEFAULT_OUTPUT_ROOT = str(Path(__file__).resolve().parent.parent)

# Paths to Save Output in React Folders.
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics"
REACT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "public" / "results"

logger = logger_utils.get_logger()


class SceneOptimizer:
    """Wrapper combining different modules to run the whole pipeline on a
    loader."""

    def __init__(
        self,
        loader: LoaderBase,
        image_pairs_generator: ImagePairsGenerator,
        correspondence_generator: CorrespondenceGeneratorBase,
        two_view_estimator: TwoViewEstimator,
        multiview_optimizer: MultiViewOptimizer,
        graph_partitioner: GraphPartitionerBase = SinglePartitioner(),
        dense_multiview_optimizer: Optional[MVSBase] = None,
        gaussian_splatting_optimizer: Optional[Any] = None,
        save_two_view_correspondences_viz: bool = False,
        save_3d_viz: bool = False,
        save_gtsfm_data: bool = True,
        pose_angular_error_thresh: float = 3,  # in degrees
        output_root: str = DEFAULT_OUTPUT_ROOT,
        output_worker: Optional[str] = None,
    ) -> None:
        self.loader = loader
        self.image_pairs_generator = image_pairs_generator
        self.correspondence_generator = correspondence_generator
        self.two_view_estimator = two_view_estimator
        self.graph_partitioner = graph_partitioner
        self.multiview_optimizer = multiview_optimizer
        self.dense_multiview_optimizer = dense_multiview_optimizer
        self.gaussian_splatting_optimizer = gaussian_splatting_optimizer

        self._save_two_view_correspondences_viz = save_two_view_correspondences_viz
        self._save_3d_viz = save_3d_viz
        self.run_dense_optimizer = self.dense_multiview_optimizer is not None
        self.run_gaussian_splatting_optimizer = self.gaussian_splatting_optimizer is not None

        self._save_gtsfm_data = save_gtsfm_data
        self._pose_angular_error_thresh = pose_angular_error_thresh
        self.output_root = Path(output_root)
        self._output_worker = output_worker
        logger.info(f"Results, plots, and metrics will be saved at {self.output_root}")

    def __repr__(self) -> str:
        """Returns string representation of class."""
        return f"""
        {self.image_pairs_generator}
        {self.correspondence_generator}
        {self.two_view_estimator}
        {self.multiview_optimizer}
        DenseMultiviewOptimizer: {self.dense_multiview_optimizer}
        GaussianSplattingOptimizer: {self.gaussian_splatting_optimizer}
        """

    def create_plot_base_path(self):
        """Create plot base path."""
        plot_base_path = self.output_root / "plots"
        os.makedirs(plot_base_path, exist_ok=True)
        return plot_base_path

    def _ensure_react_directories(self) -> None:
        """Ensure the React dashboards have dedicated output folders."""
        REACT_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        REACT_METRICS_PATH.mkdir(parents=True, exist_ok=True)

    def create_computation_graph(
        self,
        keypoints_list: list[Keypoints],
        two_view_results: AnnotatedGraph[TwoViewResult],
        num_images: int,
        one_view_data_map: dict[int, OneViewData],
        output_paths: OutputPaths,
        relative_pose_priors: AnnotatedGraph[PosePrior],
        gt_scene_mesh: Optional[Trimesh] = None,
    ) -> tuple[Delayed, list[Delayed], list[Delayed]]:
        """The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times.

        Args:
            keypoints_list: Keypoints for all images.
            two_view_results: Valid two-view results for image pairs in the cluster.
            num_images: Total number of images in the scene.
            one_view_data_map: Per-view data keyed by image index.
            output_paths: Output directories for artifacts.
            relative_pose_priors: Priors on relative poses for the cluster.
            gt_scene_mesh: Optional GT scene mesh.

        Returns:
            Tuple containing BA output, IO delayed tasks, and metrics delayed tasks.
        """

        delayed_results: list[Delayed] = []

        images = [delayed(resolve_image_future)(one_view_data_map[idx].image_future) for idx in range(num_images)]
        cameras_gt = [one_view_data_map[idx].camera_gt for idx in range(num_images)]
        gt_wTi_list = [one_view_data_map[idx].pose_gt for idx in range(num_images)]

        # Note: the MultiviewOptimizer returns BA input and BA output that are aligned to GT via Sim(3).
        (
            ba_input_graph,
            ba_output_graph,
            view_graph_two_view_reports,
            optimizer_metrics_graph,
        ) = self.multiview_optimizer.create_computation_graph(
            keypoints_list=keypoints_list,
            two_view_results=two_view_results,
            one_view_data_map=one_view_data_map,
            relative_pose_priors=relative_pose_priors,
            output_root=self.output_root,
        )
        if view_graph_two_view_reports is not None:
            two_view_reports_post_viewgraph_estimator = view_graph_two_view_reports

        # Persist all front-end metrics and their summaries.
        # TODO(akshay-krishnan): this delays saving the frontend reports until MVO has completed, not ideal.
        metrics_graph_list: list[Delayed] = []
        annotation = annotate(workers=self._output_worker) if self._output_worker else annotate()
        with annotation:
            delayed_results.append(
                delayed(save_full_frontend_metrics)(
                    {ij: r.post_isp_report for ij, r in two_view_results.items()},
                    images,
                    filename="two_view_report_{}.json".format(POST_ISP_REPORT_TAG),
                    metrics_path=output_paths.metrics,
                    plot_base_path=output_paths.plot_base,
                )
            )

            # TODO(Ayush): pass only image name instead of the whole image. And delete images from memory.
            delayed_results.append(
                delayed(save_full_frontend_metrics)(
                    two_view_reports_post_viewgraph_estimator,  # type: ignore
                    images,
                    filename="two_view_report_{}.json".format(VIEWGRAPH_REPORT_TAG),
                    metrics_path=output_paths.metrics,
                    plot_base_path=output_paths.plot_base,
                )
            )
            metrics_graph_list.append(
                delayed(two_view_estimator.aggregate_frontend_metrics)(
                    two_view_reports_post_viewgraph_estimator,  # type: ignore
                    self._pose_angular_error_thresh,
                    metric_group_name="verifier_summary_{}".format(VIEWGRAPH_REPORT_TAG),
                )
            )

        # aggregate metrics for multiview optimizer
        if optimizer_metrics_graph is not None:
            metrics_graph_list.extend(optimizer_metrics_graph)

        # Modify BA input, BA output, and GT poses to have point clouds and frustums aligned with x,y,z axes.
        ba_input_graph, ba_output_graph, gt_wTi_list = delayed(align_estimated_gtsfm_data, nout=3)(
            ba_input_graph, ba_output_graph, gt_wTi_list
        )

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
                            gt_pose_graph=gt_wTi_list,  # type: ignore
                            plot_ba_input_path=output_paths.plot_ba_input,
                            plot_results_path=output_paths.plot_results,
                        )
                    )

        if self.run_dense_optimizer and self.dense_multiview_optimizer is not None:
            img_dict_graph = delayed(get_image_dictionary)(images)
            (
                dense_points_graph,
                dense_point_colors_graph,
                densify_metrics_graph,
                downsampling_metrics_graph,
            ) = self.dense_multiview_optimizer.create_computation_graph(img_dict_graph, ba_output_graph)

            # Cast to string as Open3d cannot use PosixPath's for I/O -- only string file paths are accepted.
            annotation = annotate(workers=self._output_worker) if self._output_worker else annotate()
            with annotation:
                delayed_results.append(
                    delayed(io_utils.save_point_cloud_as_ply)(
                        save_fpath=str(output_paths.mvs_ply),
                        points=dense_points_graph,
                        rgb=dense_point_colors_graph,
                    )
                )

            # Add metrics for dense reconstruction and voxel downsampling.
            if densify_metrics_graph is not None:
                metrics_graph_list.append(densify_metrics_graph)
            if downsampling_metrics_graph is not None:
                metrics_graph_list.append(downsampling_metrics_graph)

        if self.run_gaussian_splatting_optimizer and self.gaussian_splatting_optimizer is not None:
            # this is an intentional exception from the norm to support mac implementation
            import gtsfm.splat.rendering as gtsfm_rendering

            (splats_graph, cfg_graph) = self.gaussian_splatting_optimizer.create_computation_graph(
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

        # Save metrics to JSON and generate HTML report.
        annotation = annotate(workers=self._output_worker) if self._output_worker else annotate()

        # return the entry with just the sfm result
        return ba_output_graph, delayed_results, metrics_graph_list

    def run(self, client) -> None:
        """Run the SceneOptimizer."""
        start_time = time.time()
        base_metrics_groups = []
        self._create_process_graph()
        self._ensure_react_directories()
        base_output_paths = prepare_output_paths(self.output_root, None)

        one_view_data_map = self.loader.get_one_view_data_map(client)
        num_images = len(self.loader)
        image_futures = [one_view_data_map[idx].image_future for idx in range(num_images)]
        image_fnames = [one_view_data_map[idx].image_fname for idx in range(num_images)]

        logger.info("🔥 GTSFM: Running image pair retrieval...")
        retriever_metrics, visibility_graph = self._run_retriever(client, image_futures, image_fnames)
        base_metrics_groups.append(retriever_metrics)

        logger.info("🔥 GTSFM: Running correspondence generation...")

        keypoints, putative_corr_idxs_dict, correspondence_duration_sec = self._run_correspondence_generation(
            client, visibility_graph, image_futures
        )

        logger.info("🔥 GTSFM: Running two-view estimation...")
        two_view_result_futures, tve_duration_sec = self._run_two_view_estimation(
            client,
            visibility_graph,
            keypoints,
            putative_corr_idxs_dict,
            one_view_data_map,
        )

        # Aggregate two-view metrics
        # TODO(Frank): this brings everything back ! We might not want this.
        two_view_results = client.gather(two_view_result_futures)
        base_metrics_groups.append(
            self._aggregate_two_view_metrics(
                keypoints,
                two_view_results,
                correspondence_duration_sec,
                tve_duration_sec,
                base_output_paths,
            )
        )

        logger.info("🔥 GTSFM: Partitioning the view graph...")
        assert self.graph_partitioner is not None, "Graph partitioner is not set up!"
        cluster_tree = self.graph_partitioner.run(visibility_graph)
        self.graph_partitioner.log_partition_details(cluster_tree)
        leaves = cluster_tree.leaves() if cluster_tree is not None else ()
        num_leaves = len(leaves)
        use_leaf_subdirs = num_leaves > 1

        logger.info("🔥 GTSFM: Starting to solve subgraphs...")
        futures = []
        leaf_jobs: list[tuple[int, OutputPaths]] = []
        for index, leaf in enumerate(leaves, 1):
            cluster_two_view_results = leaf.filter_annotations(two_view_results)
            if use_leaf_subdirs:
                logger.info(
                    "Creating computation graph for leaf cluster %d/%d with %d image pairs",
                    index,
                    num_leaves,
                    len(leaf.value),
                )

            if len(cluster_two_view_results) == 0:
                logger.warning(f"Skipping subgraph {index} as it has no valid two-view results.")
                continue
            output_paths = prepare_output_paths(self.output_root, index) if use_leaf_subdirs else base_output_paths
            # TODO(Frank): would be nice if relative pose prior was part of TwoViewResult
            # TODO(Frank): I think the loader should compute a Delayed dataclass, or a future

            delayed_result_io_reports = self.create_computation_graph(
                keypoints_list=keypoints,
                two_view_results=cluster_two_view_results,
                num_images=len(self.loader),
                one_view_data_map=one_view_data_map,
                relative_pose_priors=self.loader.get_relative_pose_priors(list(cluster_two_view_results.keys())),
                gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
                output_paths=output_paths,
            )
            futures.append(client.compute(delayed_result_io_reports))
            leaf_jobs.append((index, output_paths))

        logger.info("🔥 GTSFM: Running the computation graph...")
        with performance_report(filename="dask_reports/scene-optimizer.html"):
            if futures:
                results = client.gather(futures)
                for (leaf_index, output_paths), leaf_results in zip(leaf_jobs, results):
                    # leaf_results is a tuple (ba_output, io_results, mvo_metrics_groups)
                    mvo_metrics_groups = leaf_results[2]
                    if mvo_metrics_groups:
                        save_metrics_reports(mvo_metrics_groups, str(output_paths.metrics))

        # Log total time taken and save metrics report
        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info(
            "🔥 GTSFM took %.1f %s to compute sparse multi-view result.",
            duration_sec / 60 if duration_sec >= 120 else duration_sec,
            "minutes" if duration_sec >= 120 else "seconds",
        )
        total_summary_metrics = GtsfmMetricsGroup(
            "total_summary_metrics", [GtsfmMetric("total_runtime_sec", duration_sec)]
        )
        base_metrics_groups.append(total_summary_metrics)
        save_metrics_reports(base_metrics_groups, str(base_output_paths.metrics))

    def _create_process_graph(self):
        process_graph_generator = ProcessGraphGenerator()
        if isinstance(self.correspondence_generator, ImageCorrespondenceGenerator):
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

    def _run_retriever(self, client, image_futures: list[Future], image_fnames: list[str]):
        retriever_start_time = time.time()
        with performance_report(filename="dask_reports/retriever.html"):
            visibility_graph = self.image_pairs_generator.run(
                client=client,
                images=image_futures,
                image_fnames=image_fnames,
                plots_output_dir=self.create_plot_base_path(),
            )
        retriever_metrics = self.image_pairs_generator._retriever.evaluate(len(self.loader), visibility_graph)
        retriever_duration_sec = time.time() - retriever_start_time
        retriever_metrics.add_metric(GtsfmMetric("retriever_duration_sec", retriever_duration_sec))
        logger.info("🚀 Image pair retrieval took %.2f min.", retriever_duration_sec / 60.0)
        return retriever_metrics, visibility_graph

    def _run_correspondence_generation(self, client, visibility_graph, image_futures: list[Future]):
        with performance_report(filename="dask_reports/correspondence-generator.html"):
            correspondence_generation_start_time = time.time()
            (
                keypoints_list,
                putative_corr_idxs_dict,
            ) = self.correspondence_generator.generate_correspondences(
                client,
                image_futures,
                visibility_graph,
            )
            correspondence_generation_duration_sec = time.time() - correspondence_generation_start_time
        return keypoints_list, putative_corr_idxs_dict, correspondence_generation_duration_sec

    def _run_two_view_estimation(
        self,
        client,
        visibility_graph,
        keypoints_list,
        putative_corr_idxs_dict,
        one_view_data_map: dict[int, OneViewData],
    ):
        with performance_report(filename="dask_reports/two-view-estimation.html"):
            two_view_estimation_start_time = time.time()
            num_images = len(one_view_data_map)
            intrinsics = [one_view_data_map[idx].intrinsics for idx in range(num_images)]
            cameras_gt = [one_view_data_map[idx].camera_gt for idx in range(num_images)]
            # TODO(Frank):this pulls *all* results to one machine! We might not want this.
            two_view_result_futures = run_two_view_estimator_as_futures(
                client,
                self.two_view_estimator,
                keypoints_list,
                putative_corr_idxs_dict,
                intrinsics,
                self.loader.get_relative_pose_priors(visibility_graph),
                cameras_gt,
                gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
            )
            two_view_estimation_duration_sec = time.time() - two_view_estimation_start_time
        return two_view_result_futures, two_view_estimation_duration_sec

    def _maybe_save_two_view_viz(self, keypoints_list, two_view_results, plot_correspondence_path: Path):
        if self._save_two_view_correspondences_viz:
            for (i1, i2), output in two_view_results.items():
                image_i1 = self.loader.get_image(i1)
                image_i2 = self.loader.get_image(i2)
                viz_utils.save_twoview_correspondences_viz(
                    image_i1,
                    image_i2,
                    keypoints_list[i1],
                    keypoints_list[i2],
                    output.v_corr_idxs,
                    two_view_report=output.post_isp_report,
                    file_path=str(
                        plot_correspondence_path / f"{i1}_{i2}__{image_i1.file_name}_{image_i2.file_name}.jpg"
                    ),
                )

    def _aggregate_two_view_metrics(
        self,
        keypoints_list,
        two_view_results,
        correspondence_generation_duration_sec,
        two_view_estimation_duration_sec,
        output_paths: OutputPaths,
    ):
        self._maybe_save_two_view_viz(keypoints_list, two_view_results, output_paths.plot_correspondence)

        post_isp_two_view_reports_dict = {edge: output.post_isp_report for edge, output in two_view_results.items()}
        two_view_agg_metrics = two_view_estimator.aggregate_frontend_metrics(
            two_view_reports_dict=post_isp_two_view_reports_dict,
            angular_err_threshold_deg=self._pose_angular_error_thresh,
            metric_group_name="verifier_summary_{}".format(two_view_estimator.POST_ISP_REPORT_TAG),
        )
        two_view_agg_metrics.add_metric(
            GtsfmMetric("total_correspondence_generation_duration_sec", correspondence_generation_duration_sec)
        )
        two_view_agg_metrics.add_metric(
            GtsfmMetric("total_two_view_estimation_duration_sec", two_view_estimation_duration_sec)
        )
        return two_view_agg_metrics


def resolve_image_future(image_future: Future | Image) -> Image:
    """Materialize an image Future to an Image instance."""
    if isinstance(image_future, Future):
        return image_future.result()
    return image_future


def get_image_dictionary(image_list: list[Image]) -> dict[int, Image]:
    """Convert a list of images to the MVS input format."""
    img_dict = {i: img for i, img in enumerate(image_list)}
    return img_dict


def align_estimated_gtsfm_data(
    ba_input: GtsfmData, ba_output: GtsfmData, gt_wTi_list: list[Optional[Pose3]]
) -> tuple[GtsfmData, GtsfmData, list[Optional[Pose3]]]:
    """First aligns ba_input and ba_output to gt_wTi_list using a Sim3 transformation, then aligns them all to the
    X, Y, Z axes via another Sim3 global transformation.

    Args:
        ba_input: GtsfmData input to bundle adjustment.
        ba_output: GtsfmData output from bundle adjustment.
        gt_pose_graph: list of GT camera poses.

    Returns:
        Updated ba_input GtsfmData object aligned to axes.
        Updated ba_output GtsfmData object aligned to axes.
        Updated gt_pose_graph with GT poses aligned to axes.
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
    """Visualizes GtsfmData & camera poses before and after bundle adjustment using Matplotlib.

    Accepts delayed GtsfmData before and after bundle adjustment, along with GT poses,
    saves them and returns a delayed object.

    Args:
        ba_input_graph: Delayed GtsfmData input to bundle adjustment.
        ba_output_graph: Delayed GtsfmData output from bundle adjustment.
        gt_pose_graph: Delayed ground truth poses.
        plot_ba_input_path: Path to directory where visualizations of bundle adjustment input data will be saved.
        plot_results_path: Path to directory where visualizations of bundle adjustment output data will be saved.

    Returns:
        A list of Delayed objects after saving the different visualizations.
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

    Args:
        gtsfm_data: GtsfmData object with estimated camera poses and tracks.
        cameras_gt: list of GT cameras.

    Returns:
        GtsfmData object with GT camera poses and estimated tracks.
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

    Args:
        images: Input images.
        ba_input_data: GtsfmData input to bundle adjustment.
        ba_output_data: GtsfmData output to bundle adjustment.
        results_path: Path to directory where GTSFM results will be saved.
    """
    logger.info("Saving GtsfmData to %s", results_path)
    start_time = time.time()

    output_dir = results_path
    # Save the input to Bundle Adjustment (from data association).
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
    # We use the estimated tracks here, with ground truth camera poses.
    gt_gtsfm_data = get_gtsfm_data_with_gt_cameras_and_est_tracks(cameras_gt, ba_output_data)

    io_utils.export_model_as_colmap_text(
        gtsfm_data=gt_gtsfm_data,
        images=images,
        save_dir=os.path.join(output_dir, "ba_output_gt"),
    )

    # Delete old version of React results directory.
    shutil.rmtree(REACT_RESULTS_PATH)
    # Save a duplicate copy of the directory in REACT_RESULTS_PATH.
    shutil.copytree(src=results_path, dst=REACT_RESULTS_PATH)

    end_time = time.time()
    duration_sec = end_time - start_time
    logger.info("🚀 GtsfmData I/O took %.2f min.", duration_sec / 60.0)


def save_full_frontend_metrics(
    two_view_report_dict: AnnotatedGraph[TwoViewEstimationReport],
    images: list[Image],
    filename: str,
    metrics_path: Path,
    plot_base_path: Path,
) -> None:
    """Converts the TwoViewEstimationReports for all image pairs to a dict and saves it as JSON.

    Args:
        two_view_report_dict: Front-end metrics for pairs of images.
        images: list of all images for this scene, in order of image/frame index.
        filename: File name to use when saving report to JSON.
        metrics_path: Path to directory where metrics will be saved.
        plot_base_path: Path to directory where plots will be saved.
    """
    metrics_list = two_view_estimator.get_two_view_reports_summary(two_view_report_dict, images)

    io_utils.save_json_file(os.path.join(metrics_path, filename), metrics_list)

    # Save duplicate copy of 'frontend_full.json' within React Folder.
    io_utils.save_json_file(os.path.join(REACT_METRICS_PATH, filename), metrics_list)

    # All retrieval metrics need GT, no need to save them if GT is not available.
    gt_available = any([report.R_error_deg is not None for report in two_view_report_dict.values()])

    if "VIEWGRAPH_2VIEW_REPORT" in filename and gt_available:
        # must come after two-view report file is written to disk in the Dask dependency graph.
        _save_retrieval_two_view_metrics(metrics_path, plot_base_path)


def _save_retrieval_two_view_metrics(metrics_path: Path, plot_base_path: Path) -> None:
    """Compare 2-view similarity scores with their 2-view pose errors after viewgraph estimation."""
    sim_fpath = plot_base_path / "netvlad_similarity_matrix.txt"
    if not sim_fpath.exists():
        logger.warning("NetVLAD similarity matrix not found at %s. Skipping retrieval metrics." % sim_fpath)
        return

    sim = np.loadtxt(str(sim_fpath), delimiter=",")
    json_data = io_utils.read_json_file(metrics_path / "two_view_report_VIEWGRAPH_2VIEW_REPORT.json")

    sim_scores = []
    R_errors = []
    U_errors = []

    for entry in json_data:
        i1 = entry["i1"]
        i2 = entry["i2"]
        R_error = entry["rotation_angular_error"]
        U_error = entry["translation_angular_error"]
        if R_error is None or U_error is None:
            continue
        sim_score = sim[i1, i2]

        sim_scores.append(sim_score)
        R_errors.append(R_error)
        U_errors.append(U_error)

    plt.scatter(sim_scores, R_errors, 10, color="r", marker=".")
    plt.xlabel("Similarity score")
    plt.ylabel("Rotation error w.r.t. GT (deg.)")
    plt.savefig(os.path.join(plot_base_path, "gt_rot_error_vs_similarity_score.jpg"), dpi=500)
    plt.close("all")

    plt.scatter(sim_scores, U_errors, 10, color="r", marker=".")
    plt.xlabel("Similarity score")
    plt.ylabel("Translation direction error w.r.t. GT (deg.)")
    plt.savefig(os.path.join(plot_base_path, "gt_trans_error_vs_similarity_score.jpg"), dpi=500)
    plt.close("all")

    pose_errors = np.maximum(np.array(R_errors), np.array(U_errors))
    plt.scatter(sim_scores, pose_errors, 10, color="r", marker=".")
    plt.xlabel("Similarity score")
    plt.ylabel("Pose error w.r.t. GT (deg.)")
    plt.savefig(os.path.join(plot_base_path, "gt_pose_error_vs_similarity_score.jpg"), dpi=500)
    plt.close("all")


def save_metrics_reports(metrics_group_list: list[GtsfmMetricsGroup], metrics_path: str) -> None:
    """Saves metrics to JSON and HTML report.

    Args:
        metrics_graph: list of GtsfmMetricsGroup from different modules wrapped as Delayed.
        metrics_path: Path to directory where computed metrics will be saved.
    """

    # Save metrics to JSON
    metrics_utils.save_metrics_as_json(metrics_group_list, metrics_path)
    metrics_utils.save_metrics_as_json(metrics_group_list, str(REACT_METRICS_PATH))

    metrics_report.generate_metrics_report_html(
        metrics_group_list, os.path.join(metrics_path, "gtsfm_metrics_report.html"), None
    )
