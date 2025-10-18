"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""

import time
from pathlib import Path
from typing import Optional

import matplotlib
from dask.distributed import Future, performance_report

import gtsfm.two_view_estimator as two_view_estimator
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.viz as viz_utils
from gtsfm.cluster_optimizer import REACT_METRICS_PATH, REACT_RESULTS_PATH, ClusterOptimizer, save_metrics_reports
from gtsfm.common.outputs import OutputPaths, prepare_output_paths
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.graph_partitioner.single_partitioner import SinglePartitioner
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.image_pairs_generator import ImagePairsGenerator
from gtsfm.two_view_estimator import TwoViewEstimator, run_two_view_estimator_as_futures
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator

# Set matplotlib backend to "Agg" (Anti-Grain Geometry) for headless rendering
# This must be called before importing pyplot or any other matplotlib modules
# "Agg" is a non-interactive backend that renders to files without requiring a display
matplotlib.use("Agg")

DEFAULT_OUTPUT_ROOT = str(Path(__file__).resolve().parent.parent)

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
        cluster_optimizer: ClusterOptimizer,
        graph_partitioner: GraphPartitionerBase = SinglePartitioner(),
        save_two_view_correspondences_viz: bool = False,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        output_worker: Optional[str] = None,
    ) -> None:
        self.loader = loader
        self.image_pairs_generator = image_pairs_generator
        self.correspondence_generator = correspondence_generator
        self.two_view_estimator = two_view_estimator
        self.graph_partitioner = graph_partitioner
        self.cluster_optimizer = cluster_optimizer

        self._save_two_view_correspondences_viz = save_two_view_correspondences_viz
        self.output_root = Path(output_root)
        if output_worker is not None:
            self.cluster_optimizer._output_worker = output_worker
        logger.info(f"Results, plots, and metrics will be saved at {self.output_root}")

    def __repr__(self) -> str:
        """Returns string representation of class."""
        return f"""
        {self.image_pairs_generator}
        {self.correspondence_generator}
        {self.two_view_estimator}
        {self.cluster_optimizer}
        """

    def create_plot_base_path(self):
        """Create plot base path."""
        plot_base_path = self.output_root / "plots"
        plot_base_path.mkdir(parents=True, exist_ok=True)
        return plot_base_path

    def _ensure_react_directories(self) -> None:
        """Ensure the React dashboards have dedicated output folders."""
        REACT_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        REACT_METRICS_PATH.mkdir(parents=True, exist_ok=True)

    def run(self, client) -> None:
        """Run the SceneOptimizer."""
        start_time = time.time()
        base_metrics_groups = []
        self._create_process_graph()
        self._ensure_react_directories()
        base_output_paths = prepare_output_paths(self.output_root, None)

        logger.info("🔥 GTSFM: Running image pair retrieval...")
        retriever_metrics, visibility_graph, image_futures = self._run_retriever(client)
        base_metrics_groups.append(retriever_metrics)

        logger.info("🔥 GTSFM: Running correspondence generation...")
        keypoints, putative_corr_idxs_dict, correspondence_duration_sec = self._run_correspondence_generation(
            client, visibility_graph, image_futures
        )

        logger.info("🔥 GTSFM: Running two-view estimation...")
        one_view_data_dict = self.loader.get_one_view_data_dict(client)
        with performance_report(filename="dask_reports/two-view-estimation.html"):
            two_view_estimation_start_time = time.time()
            two_view_result_futures = run_two_view_estimator_as_futures(
                client,
                self.two_view_estimator,
                keypoints,
                putative_corr_idxs_dict,
                self.loader.get_relative_pose_priors(visibility_graph),
                gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
                one_view_data_dict=one_view_data_dict,
            )
            tve_duration_sec = time.time() - two_view_estimation_start_time

        # Aggregate two-view metrics
        # TODO(Frank): this brings everything back ! We might not want this.
        all_two_view_results = client.gather(two_view_result_futures)
        two_view_results = {edge: tvr for edge, tvr in all_two_view_results.items() if tvr.valid()}
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
        leaves = tuple(cluster_tree.leaves()) if cluster_tree is not None else ()
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
                logger.warning("Skipping subgraph %d as it has no valid two-view results.", index)
                continue

            output_paths = prepare_output_paths(self.output_root, index) if use_leaf_subdirs else base_output_paths

            delayed_result_io_reports = self.cluster_optimizer.create_computation_graph(
                keypoints_list=keypoints,
                two_view_results=cluster_two_view_results,
                num_images=len(self.loader),
                one_view_data_dict=one_view_data_dict,
                output_paths=output_paths,
                relative_pose_priors=self.loader.get_relative_pose_priors(list(cluster_two_view_results.keys())),
                loader=self.loader,
                output_root=self.output_root,
            )
            futures.append(client.compute(delayed_result_io_reports))
            leaf_jobs.append((index, output_paths))

        logger.info("🔥 GTSFM: Running the computation graph...")
        mvo_metrics_groups_by_leaf: dict[int, list[GtsfmMetricsGroup]] = {}
        with performance_report(filename="dask_reports/scene-optimizer.html"):
            if futures:
                results = client.gather(futures)
                for (leaf_index, output_paths), leaf_results in zip(leaf_jobs, results):
                    mvo_metrics_groups = leaf_results[2]
                    if not mvo_metrics_groups:
                        continue
                    if use_leaf_subdirs:
                        save_metrics_reports(mvo_metrics_groups, str(output_paths.metrics))
                    else:
                        mvo_metrics_groups_by_leaf[leaf_index] = mvo_metrics_groups

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

        # For single cluster runs, we may have MVO metrics to add to the base report.
        if not use_leaf_subdirs:
            for leaf_index in mvo_metrics_groups_by_leaf:
                base_metrics_groups.extend(mvo_metrics_groups_by_leaf[leaf_index])

        save_metrics_reports(base_metrics_groups, str(base_output_paths.metrics))

    def _create_process_graph(self):
        process_graph_generator = ProcessGraphGenerator()
        if isinstance(self.correspondence_generator, ImageCorrespondenceGenerator):
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

    def _run_retriever(self, client) -> tuple[GtsfmMetricsGroup, VisibilityGraph, list[Future]]:
        retriever_start_time = time.time()
        image_futures = self.loader.get_all_images_as_futures(client)
        image_fnames = self.loader.image_filenames()

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
        return retriever_metrics, visibility_graph, image_futures

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
            angular_err_threshold_deg=self.cluster_optimizer.pose_angular_error_thresh,
            metric_group_name="verifier_summary_{}".format(two_view_estimator.POST_ISP_REPORT_TAG),
        )
        two_view_agg_metrics.add_metric(
            GtsfmMetric("total_correspondence_generation_duration_sec", correspondence_generation_duration_sec)
        )
        two_view_agg_metrics.add_metric(
            GtsfmMetric("total_two_view_estimation_duration_sec", two_view_estimation_duration_sec)
        )
        return two_view_agg_metrics
