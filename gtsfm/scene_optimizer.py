"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dask.base import annotate, compute
from dask.delayed import Delayed, delayed
from dask.distributed import performance_report
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
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.graph_partitioner.single_partition import SinglePartition
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.multi_view_optimizer import MultiViewOptimizer
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
from gtsfm.utils.subgraph_utils import group_results_by_subgraph

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
        dense_multiview_optimizer: Optional[MVSBase] = None,
        gaussian_splatting_optimizer: Optional[Any] = None,
        save_two_view_correspondences_viz: bool = False,
        save_3d_viz: bool = False,
        save_gtsfm_data: bool = True,
        pose_angular_error_thresh: float = 3,  # in degrees
        output_root: str = DEFAULT_OUTPUT_ROOT,
        output_worker: Optional[str] = None,
        graph_partitioner: GraphPartitionerBase = SinglePartition(),
    ) -> None:
        self.loader = loader
        self.image_pairs_generator = image_pairs_generator
        self.correspondence_generator = correspondence_generator
        self.two_view_estimator = two_view_estimator
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
        self.graph_partitioner = graph_partitioner

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

    def create_output_directories(self, partition_index: Optional[int]) -> None:
        """Create various output directories for GTSFM results, metrics, and plots."""
        # Construct subfolder if partitioned
        partition_folder = f"partition_{partition_index}" if partition_index is not None else ""

        # Base paths
        self._plot_base_path = self.output_root / "plots" / partition_folder
        self._metrics_path = self.output_root / "result_metrics" / partition_folder
        self._results_path = self.output_root / "results" / partition_folder

        # plot paths
        self._plot_correspondence_path = self._plot_base_path / "correspondences"
        self._plot_ba_input_path = self._plot_base_path / "ba_input"
        self._plot_results_path = self._plot_base_path / "results"
        self._mvs_ply_save_fpath = self._results_path / "mvs_output" / "dense_point_cloud.ply"

        self._gs_save_path = self._results_path / "gs_output"
        self._interpolated_video_fpath = self._results_path / "gs_output" / "interpolated_path.mp4"

        # make directories for persisting data
        os.makedirs(self._plot_base_path, exist_ok=True)
        os.makedirs(self._metrics_path, exist_ok=True)
        os.makedirs(self._results_path, exist_ok=True)

        os.makedirs(self._plot_correspondence_path, exist_ok=True)
        os.makedirs(self._plot_ba_input_path, exist_ok=True)
        os.makedirs(self._plot_results_path, exist_ok=True)

        os.makedirs(self._gs_save_path, exist_ok=True)

        # Save duplicate directories within React folders.
        os.makedirs(REACT_RESULTS_PATH, exist_ok=True)
        os.makedirs(REACT_METRICS_PATH, exist_ok=True)

    def create_computation_graph(
        self,
        keypoints_list: List[Keypoints],
        two_view_results: AnnotatedGraph[TwoViewResult],
        num_images: int,
        images: List[Delayed],
        camera_intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: AnnotatedGraph[PosePrior],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        gt_wTi_list: List[Optional[Pose3]],
        gt_scene_mesh: Optional[Trimesh] = None,
    ) -> Tuple[Delayed, List[Delayed], List[Delayed]]:
        """The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times."""
        logger.info(f"Results, plots, and metrics will be saved at {self.output_root}")

        delayed_results: List[Delayed] = []

        # Note: the MultiviewOptimizer returns BA input and BA output that are aligned to GT via Sim(3).
        (
            ba_input_graph,
            ba_output_graph,
            view_graph_two_view_reports,
            optimizer_metrics_graph,
        ) = self.multiview_optimizer.create_computation_graph(
            images=images,
            num_images=num_images,
            keypoints_list=keypoints_list,
            two_view_results=two_view_results,
            all_intrinsics=camera_intrinsics,
            absolute_pose_priors=absolute_pose_priors,
            relative_pose_priors=relative_pose_priors,
            cameras_gt=cameras_gt,
            gt_wTi_list=gt_wTi_list,
            output_root=self.output_root,
        )
        if view_graph_two_view_reports is not None:
            two_view_reports_post_viewgraph_estimator = view_graph_two_view_reports

        # Persist all front-end metrics and their summaries.
        # TODO(akshay-krishnan): this delays saving the frontend reports until MVO has completed, not ideal.
        metrics_graph_list: List[Delayed] = []
        annotation = annotate(workers=self._output_worker) if self._output_worker else annotate()
        with annotation:
            delayed_results.append(
                delayed(save_full_frontend_metrics)(
                    {ij: r.post_isp_report for ij, r in two_view_results.items()},
                    images,
                    filename="two_view_report_{}.json".format(POST_ISP_REPORT_TAG),
                    metrics_path=self._metrics_path,
                    plot_base_path=self._plot_base_path,
                )
            )

            # TODO(Ayush): pass only image name instead of the whole image. And delete images from memory.
            delayed_results.append(
                delayed(save_full_frontend_metrics)(
                    two_view_reports_post_viewgraph_estimator,  # type: ignore
                    images,
                    filename="two_view_report_{}.json".format(VIEWGRAPH_REPORT_TAG),
                    metrics_path=self._metrics_path,
                    plot_base_path=self._plot_base_path,
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
                        results_path=self._results_path,
                        cameras_gt=cameras_gt,
                    )
                )
                if self._save_3d_viz:
                    delayed_results.extend(
                        save_matplotlib_visualizations(
                            aligned_ba_input_graph=ba_input_graph,
                            aligned_ba_output_graph=ba_output_graph,
                            gt_pose_graph=gt_wTi_list,  # type: ignore
                            plot_ba_input_path=self._plot_ba_input_path,
                            plot_results_path=self._plot_results_path,
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
                        save_fpath=str(self._mvs_ply_save_fpath),
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

            img_dict_graph = delayed(get_image_dictionary)(images)
            (splats_graph, cfg_graph) = self.gaussian_splatting_optimizer.create_computation_graph(
                img_dict_graph, ba_output_graph
            )

            annotation = annotate(workers=self._output_worker) if self._output_worker else annotate()
            with annotation:
                delayed_results.append(
                    delayed(gtsfm_rendering.save_splats)(save_path=str(self._gs_save_path), splats=splats_graph)
                )
                delayed_results.append(
                    delayed(gtsfm_rendering.generate_interpolated_video)(
                        images_graph=img_dict_graph,
                        sfm_result_graph=ba_output_graph,
                        cfg_result_graph=cfg_graph,
                        splats_graph=splats_graph,
                        video_fpath=self._interpolated_video_fpath,
                    )
                )

        # Save metrics to JSON and generate HTML report.
        annotation = annotate(workers=self._output_worker) if self._output_worker else annotate()

        # return the entry with just the sfm result
        return ba_output_graph, delayed_results, metrics_graph_list

    def run(self, client) -> GtsfmData:
        """Run the SceneOptimizer."""
        start_time = time.time()
        all_metrics_groups = []
        self._create_process_graph()

        logger.info("🔥 GTSFM: Running image pair retrieval...")
        retriever_metrics, visibility_graph = self._run_retriever(client)
        all_metrics_groups.append(retriever_metrics)

        logger.info("🔥 GTSFM: Running correspondence generation...")
        maybe_intrinsics, intrinsics = self._get_intrinsics_or_raise()
        keypoints, putative_corr_idxs_dict, correspondence_duration_sec = self._run_correspondence_generation(
            client, visibility_graph
        )

        logger.info("🔥 GTSFM: Running two-view estimation...")
        two_view_results, tve_duration_sec = self._run_two_view_estimation(
            client, visibility_graph, keypoints, putative_corr_idxs_dict, intrinsics
        )

        # Aggregate two-view metrics
        all_metrics_groups.append(
            self._aggregate_two_view_metrics(keypoints, two_view_results, correspondence_duration_sec, tve_duration_sec)
        )

        logger.info("🔥 GTSFM: Partitioning the view graph...")
        subgraph_two_view_results = self._partition_view_graph(visibility_graph, two_view_results)

        logger.info("🔥 GTSFM: Create back-end computation subgraphs...")
        all_delayed_sfm_results = []
        all_delayed_io = []
        all_delayed_mvo_metrics_groups = []
        for idx, subgraph_two_view_results in enumerate(subgraph_two_view_results):
            delayed_sfm_result, delayed_io, delayed_mvo_metrics_groups = self._process_subgraph(
                idx, subgraph_two_view_results, keypoints, maybe_intrinsics, len(subgraph_two_view_results)
            )
            if delayed_sfm_result is not None:
                all_delayed_sfm_results.append(delayed_sfm_result)
            all_delayed_io.extend(delayed_io)
            all_delayed_mvo_metrics_groups.extend(delayed_mvo_metrics_groups)

        logger.info("🔥 GTSFM: Starting distributed computation with Dask...")
        with performance_report(filename="dask_reports/scene-optimizer.html"):
            if all_delayed_sfm_results:
                results = compute(*all_delayed_sfm_results, *all_delayed_io, *all_delayed_mvo_metrics_groups)
                sfm_results = results[: len(all_delayed_sfm_results)]
                other_results = results[len(all_delayed_sfm_results) :]  # noqa: E203
                mvo_metrics_groups = [x for x in other_results if isinstance(x, GtsfmMetricsGroup)]
                all_metrics_groups.extend(mvo_metrics_groups)
                sfm_result = next((r for r in sfm_results if r is not None), None)
            else:
                sfm_result = None

        # Log total time taken and save metrics report
        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("🔥 GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)
        total_summary_metrics = GtsfmMetricsGroup(
            "total_summary_metrics", [GtsfmMetric("total_runtime_sec", duration_sec)]
        )
        all_metrics_groups.append(total_summary_metrics)
        save_metrics_reports(all_metrics_groups, os.path.join(self.output_root, "result_metrics"))

        return sfm_result  # type: ignore

    def _create_process_graph(self):
        process_graph_generator = ProcessGraphGenerator()
        if isinstance(self.correspondence_generator, ImageCorrespondenceGenerator):
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

    def _run_retriever(self, client):
        retriever_start_time = time.time()
        with performance_report(filename="dask_reports/retriever.html"):
            visibility_graph = self.image_pairs_generator.run(
                client=client,
                images=self.loader.get_all_images_as_futures(client),
                image_fnames=self.loader.image_filenames(),
                plots_output_dir=self.create_plot_base_path(),
            )
        retriever_metrics = self.image_pairs_generator._retriever.evaluate(len(self.loader), visibility_graph)
        retriever_duration_sec = time.time() - retriever_start_time
        retriever_metrics.add_metric(GtsfmMetric("retriever_duration_sec", retriever_duration_sec))
        logger.info("🚀 Image pair retrieval took %.2f min.", retriever_duration_sec / 60.0)
        return retriever_metrics, visibility_graph

    def _get_intrinsics_or_raise(self):
        maybe_intrinsics = self.loader.get_all_intrinsics()
        # Check if maybe_intrinsics has any None values
        if any(intrinsic is None for intrinsic in maybe_intrinsics):
            raise ValueError("Some intrinsics are None. Please ensure all intrinsics are provided.")

        # If all intrinsics are valid, cast them to the correct type
        intrinsics: list[CALIBRATION_TYPE] = maybe_intrinsics  # type: ignore
        return maybe_intrinsics, intrinsics

    def _run_correspondence_generation(self, client, visibility_graph):
        with performance_report(filename="dask_reports/correspondence-generator.html"):
            correspondence_generation_start_time = time.time()
            (
                keypoints_list,
                putative_corr_idxs_dict,
            ) = self.correspondence_generator.generate_correspondences(
                client,
                self.loader.get_all_images_as_futures(client),
                visibility_graph,
            )
            correspondence_generation_duration_sec = time.time() - correspondence_generation_start_time
        return keypoints_list, putative_corr_idxs_dict, correspondence_generation_duration_sec

    def _run_two_view_estimation(self, client, visibility_graph, keypoints_list, putative_corr_idxs_dict, intrinsics):
        with performance_report(filename="dask_reports/two-view-estimation.html"):
            two_view_estimation_start_time = time.time()
            # TODO(Frank):this pulls *all* results to one machine! We might not want this.
            all_two_view_results = run_two_view_estimator_as_futures(
                client,
                self.two_view_estimator,
                keypoints_list,
                putative_corr_idxs_dict,
                intrinsics,
                self.loader.get_relative_pose_priors(visibility_graph),
                self.loader.get_gt_cameras(),
                gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
            )
            two_view_estimation_duration_sec = time.time() - two_view_estimation_start_time
        # TODO(Frank): We might not be able to do this in a distributed manner
        two_view_results = {edge: tvr for edge, tvr in all_two_view_results.items() if tvr.valid()}
        return two_view_results, two_view_estimation_duration_sec

    def _maybe_save_two_view_viz(self, keypoints_list, two_view_results):
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
                    file_path=os.path.join(
                        self._plot_correspondence_path,
                        f"{i1}_{i2}__{image_i1.file_name}_{image_i2.file_name}.jpg",
                    ),
                )

    def _aggregate_two_view_metrics(
        self, keypoints_list, two_view_results, correspondence_generation_duration_sec, two_view_estimation_duration_sec
    ):
        self._maybe_save_two_view_viz(keypoints_list, two_view_results)

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

    def _partition_view_graph(self, visibility_graph, two_view_results):
        assert self.graph_partitioner is not None, "Graph partitioner is not set up!"
        subgraphs = self.graph_partitioner.run(visibility_graph)
        if len(subgraphs) == 1:
            # single partition
            self.create_output_directories(None)
            return [two_view_results]
        else:
            logger.info("Partitioned into %d subgraphs", len(subgraphs))
            # Group results by subgraph
            return group_results_by_subgraph(two_view_results, subgraphs)

    def _process_subgraph(self, idx, subgraph_two_view_results, keypoints_list, maybe_intrinsics, num_subgraphs):
        logger.info(
            "Creating computation graph for subgraph %d / %d with %d image pairs",
            idx + 1,
            num_subgraphs,
            len(subgraph_two_view_results),
        )
        if num_subgraphs > 1:
            self.create_output_directories(idx + 1)

        if len(subgraph_two_view_results) > 0:
            # TODO(Frank): would be nice if relative pose prior was part of TwoViewResult
            # TODO(Frank): I think the loader should compute a Delayed dataclass, or a future

            return self.create_computation_graph(
                keypoints_list=keypoints_list,
                two_view_results=subgraph_two_view_results,
                num_images=len(self.loader),
                images=self.loader.create_computation_graph_for_images(),
                camera_intrinsics=maybe_intrinsics,  # TODO(Frank): really? None is allowed?
                relative_pose_priors=self.loader.get_relative_pose_priors(list(subgraph_two_view_results.keys())),
                absolute_pose_priors=self.loader.get_absolute_pose_priors(),
                cameras_gt=self.loader.get_gt_cameras(),
                gt_wTi_list=self.loader.get_gt_poses(),
                gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
            )
        else:
            logger.warning(f"Skipping subgraph {idx+1} as it has no valid two-view results.")
            return None, [], []


def get_image_dictionary(image_list: List[Image]) -> Dict[int, Image]:
    """Convert a list of images to the MVS input format."""
    img_dict = {i: img for i, img in enumerate(image_list)}
    return img_dict


def align_estimated_gtsfm_data(
    ba_input: GtsfmData, ba_output: GtsfmData, gt_wTi_list: List[Optional[Pose3]]
) -> Tuple[GtsfmData, GtsfmData, List[Optional[Pose3]]]:
    """First aligns ba_input and ba_output to gt_wTi_list using a Sim3 transformation, then aligns them all to the
    X, Y, Z axes via another Sim3 global transformation.

    Args:
        ba_input: GtsfmData input to bundle adjustment.
        ba_output: GtsfmData output from bundle adjustment.
        gt_pose_graph: List of GT camera poses.

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
    gt_pose_graph: List[Optional[Delayed]],
    plot_ba_input_path: Path,
    plot_results_path: Path,
) -> List[Delayed]:
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
    cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
    ba_output: GtsfmData,
) -> GtsfmData:
    """Creates GtsfmData object with GT camera poses and estimated tracks.

    Args:
        gtsfm_data: GtsfmData object with estimated camera poses and tracks.
        cameras_gt: List of GT cameras.

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
    images: List[Image],
    ba_input_data: GtsfmData,
    ba_output_data: GtsfmData,
    results_path: Path,
    cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
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
    images: List[Image],
    filename: str,
    metrics_path: Path,
    plot_base_path: Path,
) -> None:
    """Converts the TwoViewEstimationReports for all image pairs to a Dict and saves it as JSON.

    Args:
        two_view_report_dict: Front-end metrics for pairs of images.
        images: List of all images for this scene, in order of image/frame index.
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
