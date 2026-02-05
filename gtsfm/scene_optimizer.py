"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeVar, cast

import matplotlib
from dask.delayed import delayed
from dask.distributed import Client, Future, performance_report

import gtsfm.utils.logger as logger_utils
from gtsfm import cluster_merging
from gtsfm.cluster_optimizer import Base, save_metrics_reports
from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterContext
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.outputs import OutputPaths, cluster_label, prepare_output_paths
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.evaluation.retrieval_metrics import save_retrieval_two_view_metrics
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.graph_partitioner.single_partitioner import SinglePartitioner
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.products.edge_quality import EdgeQualityGraph
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.image_pairs_generator import ImagePairsGenerator
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator
from gtsfm.utils.edge_quality import aggregate_edge_quality, export_edge_quality_to_json, identify_bad_edges
from gtsfm.utils.tree import PreOrderIter
from gtsfm.utils.tree_dask import submit_tree_map_with_children

# Set matplotlib backend to "Agg" (Anti-Grain Geometry) for headless rendering
# This must be called before importing pyplot or any other matplotlib modules
# "Agg" is a non-interactive backend that renders to files without requiring a display
matplotlib.use("Agg")

DEFAULT_OUTPUT_ROOT = str(Path(__file__).resolve().parent.parent)

logger = logger_utils.get_logger()
T = TypeVar("T")


@dataclass(frozen=True)
class ClusterExecutionHandles:
    """Futures tracking the execution of a single cluster optimization."""

    reconstruction: Future  # Optional[GtsfmData]
    metrics: Future  # list[GtsfmMetricsGroup]
    io_barrier: Future  # None
    output_paths: OutputPaths
    cluster_path: tuple[int, ...]
    label: str
    edge_count: int
    edge_quality: Optional[Future] = None  # Optional[EdgeQualityGraph]


def _identity(value: T) -> T:
    """Return value unchanged. Used to seed futures without extra scheduling."""
    return value


def _empty_cluster_handles(context: ClusterContext, edge_count: int) -> ClusterExecutionHandles:
    """Create placeholder futures for clusters that were skipped."""
    client = context.client
    reconstruction: Future = client.submit(_identity, None, pure=False)
    metrics: Future = client.submit(_identity, [], pure=False)
    io_barrier: Future = client.submit(_identity, None, pure=False)
    edge_quality: Future = client.submit(_identity, {}, pure=False)
    return ClusterExecutionHandles(
        reconstruction=reconstruction,
        metrics=metrics,
        io_barrier=io_barrier,
        output_paths=context.output_paths,
        cluster_path=context.cluster_path,
        label=context.label,
        edge_count=edge_count,
        edge_quality=edge_quality,
    )


def _collect_metric_results(*results: object) -> list[GtsfmMetricsGroup]:
    """Normalize metric outputs into a flat list."""
    collected: list[GtsfmMetricsGroup] = []
    for result in results:
        if result is None:
            continue
        if isinstance(result, (list, tuple)):
            for item in result:
                if item is not None:
                    collected.append(cast(GtsfmMetricsGroup, item))
        else:
            collected.append(cast(GtsfmMetricsGroup, result))
    return collected


def _finalize_io_tasks(*_args: object) -> None:
    """Barrier task used to depend on all I/O side effects."""
    return None


class SceneOptimizer:
    """Wrapper combining different modules to run the whole pipeline on a
    loader."""

    def __init__(
        self,
        loader: LoaderBase,
        image_pairs_generator: ImagePairsGenerator,
        cluster_optimizer: Base,
        graph_partitioner: GraphPartitionerBase = SinglePartitioner(),
        output_root: str = DEFAULT_OUTPUT_ROOT,
        output_worker: Optional[str] = None,
        plot_reprojection_histograms: bool = True,
    ) -> None:
        self.loader = loader
        self.image_pairs_generator = image_pairs_generator
        self.graph_partitioner = graph_partitioner
        self.cluster_optimizer = cluster_optimizer
        self._run_bundle_adjustment_on_parent = getattr(
            self.cluster_optimizer, "run_bundle_adjustment_on_parent", True
        )
        self._plot_reprojection_histograms = getattr(
            self.cluster_optimizer, "plot_reprojection_histograms", plot_reprojection_histograms
        )
        self._drop_outlier_after_camera_merging = getattr(
            self.cluster_optimizer, "drop_outlier_after_camera_merging", True
        )
        self._drop_camera_with_no_track = getattr(self.cluster_optimizer, "drop_camera_with_no_track", True)
        self._drop_child_if_merging_fail = getattr(self.cluster_optimizer, "drop_child_if_merging_fail", True)

        self.output_root = Path(output_root)
        if output_worker is not None:
            self.cluster_optimizer._output_worker = output_worker
        logger.info(f"Results, plots, and metrics will be saved at {self.output_root}")

    def __repr__(self) -> str:
        """Returns string representation of class."""
        return f"""
        {self.image_pairs_generator}
        {self.graph_partitioner}
        {self.cluster_optimizer}
        """

    def _schedule_single_cluster(self, context: ClusterContext) -> ClusterExecutionHandles:
        """Schedule the optimizer for a single cluster and return futures tracking its execution."""
        if len(context.visibility_graph) == 0:
            logger.warning("Skipping cluster %s as it has no edges.", context.label)
            return _empty_cluster_handles(context, 0)

        logger.info(
            "Creating computation graph for cluster %s with %d visibility edges.",
            context.label,
            len(context.visibility_graph),
        )

        computation = self.cluster_optimizer.create_computation_graph(
            context=context,
        )
        if computation is None or computation.sfm_result is None:
            logger.warning("Cluster optimizer produced no result for cluster %s.", context.label)
            return _empty_cluster_handles(context, len(context.visibility_graph))

        io_graph = delayed(_finalize_io_tasks, pure=False)(*computation.io_tasks)
        metrics_graph = delayed(_collect_metric_results, pure=False)(*computation.metric_tasks)
        annotated_reconstruction = delayed(cluster_merging.annotate_scene_with_metadata, pure=False)(
            computation.sfm_result,
            context.output_paths.plots,
            context.label,
        )

        io_future: Future = context.client.compute(io_graph)  # type: ignore
        metrics_future: Future = context.client.compute(metrics_graph)  # type: ignore
        reconstruction_future: Future = context.client.compute(annotated_reconstruction)  # type: ignore

        # Compute edge quality if available
        edge_quality_future: Optional[Future] = None
        if computation.edge_quality is not None:
            edge_quality_future = context.client.compute(computation.edge_quality)  # type: ignore

        return ClusterExecutionHandles(
            reconstruction=reconstruction_future,
            metrics=metrics_future,
            io_barrier=io_future,
            output_paths=context.output_paths,
            cluster_path=context.cluster_path,
            label=context.label,
            edge_count=len(context.visibility_graph),
            edge_quality=edge_quality_future,
        )

    def run(self, client: Client) -> None:
        """Run the SceneOptimizer."""
        start_time = time.time()
        base_metrics_groups = []

        # Process Graph Generation: Visualize the process graph, which is a flow of data across GTSFM's modules.
        process_graph_generator = ProcessGraphGenerator()
        base_output_paths = prepare_output_paths(self.output_root, None)
        process_graph_generator.save_graph(str(base_output_paths.plots / "process_graph_output.svg"))

        logger.info("🔥 GTSFM: Running image pair retrieval...")
        retriever_metrics, visibility_graph = self._run_retriever(client, base_output_paths)
        base_metrics_groups.append(retriever_metrics)
        image_future_map = self.loader.get_image_futures(client)

        # Graph partitioning: Divide the visibility graph into clusters (runs eagerly, no delayed/futures).
        logger.info("🔥 GTSFM: Partitioning the view graph...")
        assert self.graph_partitioner is not None, "Graph partitioner is not set up!"
        cluster_tree = self.graph_partitioner.run(visibility_graph)
        self.graph_partitioner.log_partition_details(cluster_tree, base_output_paths)
        save_retrieval_two_view_metrics(base_output_paths)

        logger.info("🔥 GTSFM: Scheduling cluster optimizations...")
        one_view_data_dict = self.loader.get_one_view_data_dict()
        merged_scene: Optional[GtsfmData] = None

        with performance_report(filename="dask_reports/scene-optimizer.html"):
            if cluster_tree is None:
                logger.warning("No clusters generated by partitioner; skipping reconstruction and merge.")
            else:
                num_images = len(self.loader)

                def to_context(path: tuple[int, ...], visibility_graph: VisibilityGraph) -> ClusterContext:
                    output_paths = base_output_paths if len(path) == 0 else prepare_output_paths(self.output_root, path)
                    return ClusterContext(
                        client=client,
                        loader=self.loader,
                        num_images=num_images,
                        output_paths=output_paths,
                        image_future_map=image_future_map,
                        one_view_data_dict=one_view_data_dict,
                        cluster_path=path,
                        label=cluster_label(path),
                        visibility_graph=visibility_graph,
                    )

                context_tree = cluster_tree.map_with_path(to_context)

                # Runs reconstruction on each node of the VisibilityGraph (with context) tree.
                # Returns handles to various outputs: reconstruction, metrics, io_barrier etc.
                handles_tree = context_tree.map(self._schedule_single_cluster)

                # Collect and evaluate edge quality from all clusters BEFORE fold
                all_edge_quality = self._collect_and_evaluate_edge_quality(
                    handles_tree, base_output_paths, visibility_graph
                )

                # Get the reconstruction handle and run merging to get a tree of merged result handles. 
                reconstruction_tree = handles_tree.map(lambda handle: handle.reconstruction)
                cameras_gt = self.loader.get_gt_cameras()

                def merge_fn(
                    reconstruction: object, child_results: tuple[cluster_merging.MergedNodeResult, ...]
                ) -> cluster_merging.MergedNodeResult:
                    return cluster_merging.combine_results(
                        cast(Optional[GtsfmData], reconstruction),
                        child_results,
                        cameras_gt=cameras_gt,
                        run_bundle_adjustment_on_parent=self._run_bundle_adjustment_on_parent,
                        plot_reprojection_histograms=self._plot_reprojection_histograms,
                        drop_outlier_after_camera_merging=self._drop_outlier_after_camera_merging,
                        drop_camera_with_no_track=self._drop_camera_with_no_track,
                        drop_child_if_merging_fail=self._drop_child_if_merging_fail,
                        store_full_data=False,
                    )

                merged_future_tree = submit_tree_map_with_children(client, reconstruction_tree, merge_fn)
                export_tree = cluster_merging.schedule_exports(client, handles_tree, merged_future_tree)
                root_merge_future: Optional[Future] = merged_future_tree.value
                for handle_node, merged_node, export_node in zip(
                    PreOrderIter(handles_tree),
                    PreOrderIter(merged_future_tree),
                    PreOrderIter(export_tree),
                ):
                    handle = handle_node.value
                    merge_future = merged_node.value
                    export_future = export_node.value

                    metrics_groups = list(handle.metrics.result())
                    handle.io_barrier.result()
                    export_future.result()
                    if handle.cluster_path == ():
                        merged_result = merge_future.result()
                        base_metrics_groups.extend(metrics_groups)
                        base_metrics_groups.append(merged_result.metrics)
                        root_merge_future = merge_future
                    elif metrics_groups:
                        merged_result = merge_future.result()
                        metrics_groups.append(merged_result.metrics)
                        save_metrics_reports(metrics_groups, str(handle.output_paths.metrics))
                if root_merge_future is not None:
                    logger.info("🔥 GTSFM: Running cluster optimization and merging...")
                    root_merge_result = root_merge_future.result()
                    merged_scene = root_merge_result.scene

        if merged_scene is not None:
            logger.info(
                "Merged scene contains %d images and %d tracks.",
                merged_scene.number_images(),
                merged_scene.number_tracks(),
            )
        else:
            logger.warning("Merging failed, no final merged scene found.")

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

    def _collect_and_evaluate_edge_quality(
        self,
        handles_tree,
        base_output_paths: OutputPaths,
        visibility_graph: VisibilityGraph,
    ) -> EdgeQualityGraph:
        """Collect edge quality from all clusters and evaluate for bad edges.

        This is called AFTER the map step (cluster reconstructions) but BEFORE
        the fold step (merging). If bad edges are found, they are logged and
        exported for analysis.

        Args:
            handles_tree: Tree of ClusterExecutionHandles from cluster optimizations.
            base_output_paths: Output paths for saving edge quality report.
            visibility_graph: Original visibility graph for reference.

        Returns:
            Aggregated EdgeQualityGraph from all clusters.
        """
        logger.info("🔍 Collecting edge quality from cluster reconstructions...")

        # Collect edge quality from all cluster handles
        cluster_edge_qualities: list[EdgeQualityGraph] = []
        for handle_node in PreOrderIter(handles_tree):
            handle = handle_node.value
            if handle.edge_quality is not None:
                try:
                    edge_quality_result = handle.edge_quality.result()
                    if edge_quality_result:
                        cluster_edge_qualities.append(edge_quality_result)
                        logger.debug(
                            "Collected %d edge quality scores from cluster %s",
                            len(edge_quality_result),
                            handle.label,
                        )
                except Exception as exc:
                    logger.warning("Failed to collect edge quality from cluster %s: %s", handle.label, exc)

        if not cluster_edge_qualities:
            logger.warning("No edge quality data collected from any cluster.")
            return {}

        # Aggregate edge quality (handles edges appearing in multiple clusters)
        all_edge_quality = aggregate_edge_quality(cluster_edge_qualities)
        logger.info("Aggregated edge quality for %d edges.", len(all_edge_quality))

        # Identify bad edges
        bad_edges = identify_bad_edges(all_edge_quality)
        if bad_edges:
            logger.warning(
                "⚠️ Identified %d bad edges out of %d total (%.1f%%).",
                len(bad_edges),
                len(all_edge_quality),
                100.0 * len(bad_edges) / len(all_edge_quality) if all_edge_quality else 0,
            )
        else:
            logger.info("✅ All %d edges passed quality thresholds.", len(all_edge_quality))

        # Export edge quality report for debugging
        try:
            export_path = Path(base_output_paths.results) / "edge_quality_report.json"
            export_edge_quality_to_json(all_edge_quality, bad_edges, export_path)
            logger.info("Edge quality report saved to %s", export_path)
        except Exception as exc:
            logger.warning("Failed to export edge quality report: %s", exc)

        return all_edge_quality

    def _run_retriever(self, client: Client, output_paths: OutputPaths) -> tuple[GtsfmMetricsGroup, VisibilityGraph]:
        # TODO(Frank): refactor to move more of this logic into ImagePairsGenerator
        retriever_start_time = time.time()
        batch_size = self.image_pairs_generator._batch_size

        transforms = self.image_pairs_generator.get_preprocessing_transforms()

        # Image_Batch_Futures is a list of Stacked Tensors with dimension (batch_size, Channels, H, W)
        image_batch_futures = self.loader.get_all_descriptor_image_batches_as_futures(client, batch_size, *transforms)

        image_fnames = self.loader.image_filenames()

        plots_output_dir = output_paths.plots
        with performance_report(filename="dask_reports/retriever.html"):
            visibility_graph = self.image_pairs_generator.run(
                client=client,
                image_batch_futures=image_batch_futures,
                image_fnames=image_fnames,
                plots_output_dir=plots_output_dir,
            )

        retriever = self.image_pairs_generator._retriever
        try:
            retriever.save_diagnostics(
                image_fnames=image_fnames,
                pairs=visibility_graph,
                plots_output_dir=plots_output_dir,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path best-effort
            logger.warning("Failed to persist retriever diagnostics: %s", exc)

        retriever_metrics = self.image_pairs_generator._retriever.evaluate(len(self.loader), visibility_graph)
        retriever_duration_sec = time.time() - retriever_start_time
        retriever_metrics.add_metric(GtsfmMetric("retriever_duration_sec", retriever_duration_sec))
        logger.info("🚀 Image pair retrieval took %.2f min.", retriever_duration_sec / 60.0)

        return retriever_metrics, visibility_graph
