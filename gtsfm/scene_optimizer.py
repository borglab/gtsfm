"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""

import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, TypeVar, cast

import matplotlib
from dask.delayed import delayed
from dask.distributed import Client, Future, performance_report

import gtsfm.utils.logger as logger_utils
from gtsfm.cluster_optimizer import REACT_METRICS_PATH, REACT_RESULTS_PATH, Base, save_metrics_reports
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.outputs import OutputPaths, cluster_label, prepare_output_paths
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.evaluation.retrieval_metrics import save_retrieval_two_view_metrics
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.graph_partitioner.single_partitioner import SinglePartitioner
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.image_pairs_generator import ImagePairsGenerator
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator
from gtsfm.utils import align as align_utils
from gtsfm.utils.tree import PreOrderIter, Tree
from gtsfm.utils.tree_dask import submit_tree_map, submit_tree_map_with_children

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


@dataclass(frozen=True)
class ClusterNodeContext:
    """Static metadata describing a cluster tree node."""

    visibility_graph: VisibilityGraph
    output_paths: OutputPaths
    cluster_path: tuple[int, ...]
    label: str

    @property
    def is_root(self) -> bool:
        return len(self.cluster_path) == 0


def _identity(value: T) -> T:
    """Return value unchanged. Used to seed futures without extra scheduling."""
    return value


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


def _export_merged_scene(
    merged_scene: Optional[GtsfmData],
    target_dir: Path,
    image_shapes: Sequence[tuple[int, ...]],
    image_filenames: Sequence[str],
    images: Optional[Sequence[Image]] = None,
) -> None:
    """Persist a merged reconstruction to COLMAP text format."""
    if merged_scene is None:
        return

    merged_path = Path(target_dir)
    merged_path.mkdir(parents=True, exist_ok=True)
    merged_scene.export_as_colmap_text(
        merged_path,
        images=list(images) if images is not None else None,
        image_shapes=list(image_shapes),
        image_filenames=list(image_filenames),
    )


def _run_export_task(
    payload: tuple[ClusterExecutionHandles, Optional[GtsfmData]],
    image_shapes: Sequence[tuple[int, ...]],
    image_filenames: Sequence[str],
    images: Sequence[Image] | None,
) -> None:
    """Execute merged scene export on a worker."""
    handle, merged_scene = payload
    merged_dir = handle.output_paths.results / "merged"
    _export_merged_scene(merged_scene, merged_dir, image_shapes, image_filenames, images)


def _merge_cluster_results(
    current: Optional[GtsfmData], child_results: tuple[Optional[GtsfmData], ...]
) -> Optional[GtsfmData]:
    """Merge bundle adjustment outputs from child clusters into the parent result."""
    merged = current
    for child in child_results:
        if child is None:
            continue
        if merged is None:
            merged = child
            continue
        try:
            aSb = align_utils.sim3_from_Pose3_maps(merged.poses(), child.poses())
            merged = merged.merged_with(child, aSb)
        except Exception as exc:
            logger.warning("Failed to merge cluster outputs: %s", exc)
    return merged


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
    ) -> None:
        self.loader = loader
        self.image_pairs_generator = image_pairs_generator
        self.graph_partitioner = graph_partitioner
        self.cluster_optimizer = cluster_optimizer

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

    def _ensure_react_directories(self) -> None:
        """Ensure the React dashboards have dedicated output folders."""
        REACT_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        REACT_METRICS_PATH.mkdir(parents=True, exist_ok=True)

    def _save_retrieval_diagnostics(self, output_paths: OutputPaths) -> None:
        try:
            save_retrieval_two_view_metrics(output_paths.metrics, output_paths.plots)
        except Exception as exc:  # pragma: no cover - diagnostics are best-effort
            logger.debug("Skipping retrieval diagnostics for %s: %s", output_paths.plots, exc)

    def _build_cluster_context_tree(
        self, cluster_tree: Tree[VisibilityGraph], base_output_paths: OutputPaths
    ) -> Tree[ClusterNodeContext]:
        """Annotate each cluster node with static metadata required for scheduling."""

        def to_context(path: tuple[int, ...], visibility_graph: VisibilityGraph) -> ClusterNodeContext:
            output_paths = base_output_paths if len(path) == 0 else prepare_output_paths(self.output_root, path)
            return ClusterNodeContext(
                visibility_graph=visibility_graph,
                output_paths=output_paths,
                cluster_path=path,
                label=cluster_label(path),
            )

        return cluster_tree.map_with_path(to_context)

    def run(self, client: Client) -> None:
        """Run the SceneOptimizer."""
        start_time = time.time()
        base_metrics_groups = []
        process_graph_generator = ProcessGraphGenerator()
        self._ensure_react_directories()
        base_output_paths = prepare_output_paths(self.output_root, None)
        process_graph_generator.save_graph(str(base_output_paths.plots / "process_graph_output.svg"))

        logger.info("🔥 GTSFM: Running image pair retrieval...")
        retriever_metrics, visibility_graph = self._run_retriever(client, base_output_paths)
        base_metrics_groups.append(retriever_metrics)
        image_futures = self.loader.get_all_images_as_futures(client)

        logger.info("🔥 GTSFM: Partitioning the view graph...")
        assert self.graph_partitioner is not None, "Graph partitioner is not set up!"
        cluster_tree = self.graph_partitioner.run(visibility_graph)
        self.graph_partitioner.log_partition_details(cluster_tree, base_output_paths)

        logger.info("🔥 GTSFM: Scheduling cluster optimizations...")
        one_view_data_dict = self.loader.get_one_view_data_dict()
        merged_scene: Optional[GtsfmData] = None

        with performance_report(filename="dask_reports/scene-optimizer.html"):
            if cluster_tree is None:
                logger.warning("No clusters generated by partitioner; skipping reconstruction and merge.")
            else:
                context_tree = self._build_cluster_context_tree(cluster_tree, base_output_paths)
                handles_tree = self._schedule_cluster_tree(
                    client=client,
                    context_tree=context_tree,
                    num_images=len(self.loader),
                    one_view_data_dict=one_view_data_dict,
                    image_futures=image_futures,
                )
                image_shapes = self.loader.get_image_shapes()
                image_filenames = self.loader.image_filenames()
                merged_tree = self._schedule_merge_tree(client, handles_tree)
                export_tree = self._schedule_merge_exports(
                    client=client,
                    handles_tree=handles_tree,
                    merged_tree=merged_tree,
                    image_shapes=image_shapes,
                    image_filenames=image_filenames,
                    image_futures=image_futures,
                )
                root_merge_future: Optional[Future] = merged_tree.value
                for handle_node, merged_node, export_node in zip(
                    PreOrderIter(handles_tree), PreOrderIter(merged_tree), PreOrderIter(export_tree)
                ):
                    handle = handle_node.value
                    merge_future = merged_node.value
                    export_future = export_node.value
                    metrics_groups = list(handle.metrics.result())
                    handle.io_barrier.result()
                    export_future.result()
                    if handle.cluster_path == ():
                        base_metrics_groups.extend(metrics_groups)
                        root_merge_future = merge_future
                    elif metrics_groups:
                        save_metrics_reports(metrics_groups, str(handle.output_paths.metrics))
                if root_merge_future is not None:
                    logger.info("🔥 GTSFM: Running cluster optimization and merging...")
                    merged_scene = root_merge_future.result()

        if merged_scene is not None:
            logger.info(
                "Merged scene contains %d images and %d tracks.",
                merged_scene.number_images(),
                merged_scene.number_tracks(),
            )

        if not futures:
            self._save_retrieval_diagnostics(base_output_paths)

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

    def _schedule_cluster_tree(
        self,
        *,
        client: Client,
        context_tree: Tree[ClusterNodeContext],
        num_images: int,
        one_view_data_dict,
        image_futures: Sequence[Future],
    ) -> Tree[ClusterExecutionHandles]:
        """Schedule reconstruction for every cluster in the tree."""

        def schedule(context: ClusterNodeContext) -> ClusterExecutionHandles:
            return self._schedule_single_cluster(
                client=client,
                context=context,
                num_images=num_images,
                one_view_data_dict=one_view_data_dict,
                image_futures=image_futures,
            )

        return context_tree.map(schedule)

    def _schedule_single_cluster(
        self,
        *,
        client: Client,
        context: ClusterNodeContext,
        num_images: int,
        one_view_data_dict,
        image_futures: Sequence[Future],
    ) -> ClusterExecutionHandles:
        """Schedule the optimizer for a single cluster and return futures tracking its execution."""
        visibility_graph = context.visibility_graph
        output_paths = context.output_paths
        cluster_path = context.cluster_path
        label = context.label
        edge_count = len(visibility_graph)
        if edge_count == 0:
            logger.warning("Skipping cluster %s as it has no edges.", label)
            return self._empty_cluster_handles(client, context, edge_count)

        logger.info("Creating computation graph for cluster %s with %d visibility edges.", label, edge_count)

        computation = self.cluster_optimizer.create_computation_graph(
            num_images=num_images,
            one_view_data_dict=one_view_data_dict,
            output_paths=output_paths,
            loader=self.loader,
            output_root=self.output_root,
            visibility_graph=visibility_graph,
            image_futures=image_futures,
        )
        if computation is None or computation.sfm_result is None:
            logger.warning("Cluster optimizer produced no result for cluster %s.", label)
            return self._empty_cluster_handles(client, context, edge_count)

        io_graph = delayed(_finalize_io_tasks, pure=False)(*computation.io_tasks)
        metrics_graph = delayed(_collect_metric_results, pure=False)(*computation.metric_tasks)

        io_future: Future = client.compute(io_graph)  # type: ignore
        metrics_future: Future = client.compute(metrics_graph)  # type: ignore
        reconstruction_future: Future = client.compute(computation.sfm_result)  # type: ignore

        return ClusterExecutionHandles(
            reconstruction=reconstruction_future,
            metrics=metrics_future,
            io_barrier=io_future,
            output_paths=output_paths,
            cluster_path=cluster_path,
            label=label,
            edge_count=edge_count,
        )

    def _empty_cluster_handles(
        self,
        client: Client,
        context: ClusterNodeContext,
        edge_count: int,
    ) -> ClusterExecutionHandles:
        """Create placeholder futures for clusters that were skipped."""
        reconstruction: Future = client.submit(_identity, None, pure=False)
        metrics: Future = client.submit(_identity, [], pure=False)
        io_barrier: Future = client.submit(_identity, None, pure=False)
        return ClusterExecutionHandles(
            reconstruction=reconstruction,
            metrics=metrics,
            io_barrier=io_barrier,
            output_paths=context.output_paths,
            cluster_path=context.cluster_path,
            label=context.label,
            edge_count=edge_count,
        )

    def _schedule_merge_tree(self, client: Client, handles_tree: Tree[ClusterExecutionHandles]) -> Tree[Future]:
        """Create merge futures for every cluster mirroring the execution handles tree."""
        reconstruction_tree = handles_tree.map(lambda handle: handle.reconstruction)
        return submit_tree_map_with_children(client, reconstruction_tree, _merge_cluster_results)

    def _schedule_merge_exports(
        self,
        *,
        client: Client,
        handles_tree: Tree[ClusterExecutionHandles],
        merged_tree: Tree[Future],
        image_shapes: Sequence[tuple[int, ...]],
        image_filenames: Sequence[str],
        image_futures: Sequence[Future],
    ) -> Tree[Future]:
        """Schedule persistence of merged reconstructions for each cluster."""
        export_payload_tree = Tree.zip(handles_tree, merged_tree)
        export_fn = partial(
            _run_export_task,
            image_shapes=tuple(image_shapes),
            image_filenames=tuple(image_filenames),
            images=tuple(image_futures),
        )
        return submit_tree_map(client, export_payload_tree, export_fn, pure=False)
