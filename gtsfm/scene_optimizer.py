"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""

import time
from pathlib import Path
from typing import Optional

import matplotlib
from dask.distributed import Future, performance_report

import gtsfm.utils.logger as logger_utils
from gtsfm.cluster_optimizer import REACT_METRICS_PATH, REACT_RESULTS_PATH, ClusterOptimizer
from gtsfm.common.outputs import Outputs, prepare_outputs
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.graph_partitioner.single_partitioner import SinglePartitioner
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.image_pairs_generator import ImagePairsGenerator
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
        cluster_optimizer: ClusterOptimizer,
        graph_partitioner: GraphPartitionerBase = SinglePartitioner(),
        output_root: str = DEFAULT_OUTPUT_ROOT,
        output_worker: Optional[str] = None,
        metrics_enabled: bool = True,
    ) -> None:
        self.loader = loader
        self.image_pairs_generator = image_pairs_generator
        self.graph_partitioner = graph_partitioner
        self.cluster_optimizer = cluster_optimizer
        self._metrics_enabled = metrics_enabled

        self.output_root = Path(output_root)
        if output_worker is not None:
            self.cluster_optimizer._output_worker = output_worker
        if self._metrics_enabled:
            logger.info(f"Results, plots, and metrics will be saved at {self.output_root}")
        else:
            logger.info(f"Results and plots will be saved at {self.output_root} (metrics disabled)")

    def __repr__(self) -> str:
        """Returns string representation of class."""
        return f"""
        {self.image_pairs_generator}
        {self.graph_partitioner}
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
        self._create_process_graph()
        self._ensure_react_directories()
        base_output_paths = prepare_outputs(self.output_root, None, enable_metrics=self._metrics_enabled)

        logger.info("🔥 GTSFM: Running image pair retrieval...")
        visibility_graph, image_futures = self._run_retriever(client, base_output_paths)

        logger.info("🔥 GTSFM: Partitioning the view graph...")
        assert self.graph_partitioner is not None, "Graph partitioner is not set up!"
        cluster_tree = self.graph_partitioner.run(visibility_graph)
        self.graph_partitioner.log_partition_details(cluster_tree)
        leaves = tuple(cluster_tree.leaves()) if cluster_tree is not None else ()
        num_leaves = len(leaves)
        use_leaf_subdirs = num_leaves > 1

        logger.info("🔥 GTSFM: Starting to solve subgraphs...")
        futures = []
        one_view_data_dict = self.loader.get_one_view_data_dict()
        for index, leaf in enumerate(leaves, 1):
            cluster_visibility_graph = leaf.value
            if use_leaf_subdirs:
                logger.info(
                    "Creating computation graph for leaf cluster %d/%d with %d image pairs",
                    index,
                    num_leaves,
                    len(cluster_visibility_graph),
                )

            if len(cluster_visibility_graph) == 0:
                logger.warning("Skipping subgraph %d as it has no edges.", index)
                continue

            output_paths = (
                prepare_outputs(self.output_root, index, enable_metrics=self._metrics_enabled)
                if use_leaf_subdirs
                else base_output_paths
            )

            delayed_result_io_reports = self.cluster_optimizer.create_computation_graph(
                num_images=len(self.loader),
                one_view_data_dict=one_view_data_dict,
                output_paths=output_paths,
                loader=self.loader,
                output_root=self.output_root,
                visibility_graph=cluster_visibility_graph,
                image_futures=image_futures,
            )
            if delayed_result_io_reports is None:
                logger.warning("Skipping subgraph %d as it has no valid two-view results.", index)
                continue
            futures.append(client.compute(delayed_result_io_reports))

        logger.info("🔥 GTSFM: Running the computation graph...")
        with performance_report(filename="dask_reports/scene-optimizer.html"):
            if futures:
                client.gather(futures)

        # Log total time taken and persist summary metrics.
        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info(
            "🔥 GTSFM took %.1f %s to compute sparse multi-view result.",
            duration_sec / 60 if duration_sec >= 120 else duration_sec,
            "minutes" if duration_sec >= 120 else "seconds",
        )
        sink = base_output_paths.metrics_sink
        if sink is not None:
            sink.record(GtsfmMetricsGroup("total_summary_metrics", [GtsfmMetric("total_runtime_sec", duration_sec)]))

    def _create_process_graph(self):
        process_graph_generator = ProcessGraphGenerator()
        if isinstance(self.cluster_optimizer.correspondence_generator, ImageCorrespondenceGenerator):
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

    def _run_retriever(self, client, outputs: Outputs) -> tuple[VisibilityGraph, list[Future]]:
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
        retriever_duration_sec = time.time() - retriever_start_time
        self.image_pairs_generator._retriever.evaluate(
            len(self.loader),
            visibility_graph,
            outputs=outputs,
            additional_metrics=[GtsfmMetric("retriever_duration_sec", retriever_duration_sec)],
        )
        logger.info("🚀 Image pair retrieval took %.2f min.", retriever_duration_sec / 60.0)
        return visibility_graph, image_futures
