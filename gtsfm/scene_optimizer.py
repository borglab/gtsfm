"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeVar, cast

import matplotlib
import numpy as np
from dask.delayed import delayed
from dask.distributed import Client, Future, performance_report
from omegaconf import OmegaConf

import gtsfm.utils.logger as logger_utils
from gtsfm import cluster_merging
from gtsfm.cluster_merging import MergingOptions
from gtsfm.cluster_optimizer import Base, save_metrics_reports
from gtsfm.cluster_optimizer.cluster_mvo import ClusterMVO, _pad_keypoints_list
from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterContext
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.outputs import OutputPaths, cluster_label, prepare_output_paths
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.evaluation.retrieval_metrics import save_retrieval_two_view_metrics
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.graph_partitioner.single_partitioner import SinglePartitioner
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.visibility_graph import AnnotatedGraph, VisibilityGraph, visibility_graph_keys
from gtsfm.retriever.image_pairs_generator import ImagePairsGenerator
from gtsfm.two_view_estimator import TwoViewEstimator, create_v_corr_idxs_futures, create_v_corr_idxs_inline
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator
from gtsfm.utils.graph import get_nodes_in_largest_connected_component
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


def _identity(value: T) -> T:
    """Return value unchanged. Used to seed futures without extra scheduling."""
    return value


def _empty_cluster_handles(context: ClusterContext, edge_count: int) -> ClusterExecutionHandles:
    """Create placeholder futures for clusters that were skipped."""
    client = context.client
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


def verify_visibility_graph(
    client: Client,
    correspondence_generator: CorrespondenceGeneratorBase,
    two_view_estimator: TwoViewEstimator,
    loader: LoaderBase,
    image_future_map: dict[int, Future],
    one_view_data_dict: dict[int, OneViewData],
    visibility_graph: VisibilityGraph,
) -> tuple[list[Keypoints], AnnotatedGraph[np.ndarray]]:
    """Run the frontend once over the retrieval graph, returning keypoints + verified correspondences.

    Correspondence generation and two-view estimation run over ALL retrieval edges; an edge survives iff
    its ``TwoViewResult.valid()``. Only the verified correspondence indices are kept (each heavy
    ``TwoViewResult`` is dropped the moment it is reduced), and every ``run_2view`` call warms the
    two-view cache, so per-cluster frontends that run afterwards are cache hits.

    Dispatches on pool size: with multiple workers the frontend fans out across the pool and only the
    lean per-chunk ``v_corr_idxs`` sub-dicts are gathered; with at most one worker everything runs inline
    in this process, leaving no scheduler<->worker comm surface for a multi-hour run to trip over.

    Args:
        client: Dask client.
        correspondence_generator: Frontend correspondence generator (detection + matching).
        two_view_estimator: Two-view estimator applied to every pair.
        loader: Dataset loader (pose priors, GT mesh, image count).
        image_future_map: Scattered images, keyed by image index.
        one_view_data_dict: Per-image intrinsics / GT data.
        visibility_graph: Retrieval graph to verify.

    Returns:
        keypoints_list: Per-image keypoints, padded to ``len(loader)``.
        v_corr_idxs_dict: Verified correspondence indices for every surviving edge.
    """
    num_images = len(loader)
    image_futures = [image_future_map[idx] for idx in range(num_images)]
    relative_pose_priors = loader.get_relative_pose_priors(list(visibility_graph)) or {}
    gt_scene_mesh = loader.get_gt_scene_trimesh()
    try:
        num_workers = len(client.scheduler_info()["workers"])
    except Exception:
        num_workers = 1

    if num_workers <= 1:
        logger.info("🔵 [frontend] 1 worker → running the global frontend inline (in-process).")
        images = client.gather(image_futures)
        keypoints_list, putative_corr_idxs_dict, _ = ClusterMVO._run_correspondence_generator(
            correspondence_generator, list(visibility_graph), images
        )
        padded_keypoints_list = _pad_keypoints_list(keypoints_list, num_images)
        v_corr_idxs_dict = create_v_corr_idxs_inline(
            two_view_estimator,
            padded_keypoints_list,
            putative_corr_idxs_dict,
            relative_pose_priors,
            gt_scene_mesh,
            one_view_data_dict,
        )
    else:
        logger.info("🔵 [frontend] %d workers → running the global frontend in parallel.", num_workers)
        keypoints_list, putative_corr_idxs_dict = correspondence_generator.generate_correspondences(
            client, image_futures, list(visibility_graph)
        )
        padded_keypoints_list = _pad_keypoints_list(keypoints_list, num_images)
        v_corr_idxs_dict = create_v_corr_idxs_futures(
            client,
            two_view_estimator,
            padded_keypoints_list,
            putative_corr_idxs_dict,
            relative_pose_priors,
            gt_scene_mesh,
            one_view_data_dict,
        )
    return padded_keypoints_list, v_corr_idxs_dict


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
        merging_options: MergingOptions | None = None,
        # --- Bridge params ---
        bridge_min_similarity: float = 0.0,
        bridge_top_k: int = 10,
        bridge_min_component_size: int = 3,
    ) -> None:
        self.loader = loader
        self.image_pairs_generator = image_pairs_generator
        self.graph_partitioner = graph_partitioner
        self.cluster_optimizer = cluster_optimizer
        self._merging_options = merging_options or MergingOptions()
        self._bridge_min_similarity = bridge_min_similarity
        self._bridge_top_k = bridge_top_k
        self._bridge_min_component_size = bridge_min_component_size
        # Propagate metric_constructed_only to the cluster optimizer if it supports it.
        if hasattr(self.cluster_optimizer, "_metric_constructed_only"):
            setattr(self.cluster_optimizer, "_metric_constructed_only", self._merging_options.metric_constructed_only)
        elif hasattr(self.cluster_optimizer, "_optimizer") and hasattr(
            getattr(self.cluster_optimizer, "_optimizer"), "_metric_constructed_only"
        ):
            setattr(
                self.cluster_optimizer._optimizer,
                "_metric_constructed_only",
                self._merging_options.metric_constructed_only,
            )
        self._config_snapshot = None
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

        return ClusterExecutionHandles(
            reconstruction=reconstruction_future,
            metrics=metrics_future,
            io_barrier=io_future,
            output_paths=context.output_paths,
            cluster_path=context.cluster_path,
            label=context.label,
            edge_count=len(context.visibility_graph),
        )

    def run(self, client: Client) -> None:
        """Run the SceneOptimizer."""
        start_time = time.time()
        base_metrics_groups = []

        # Process Graph Generation: Visualize the process graph, which is a flow of data across GTSFM's modules.
        process_graph_generator = ProcessGraphGenerator()
        base_output_paths = prepare_output_paths(self.output_root, None)
        config_snapshot = self._config_snapshot
        if config_snapshot is not None:
            config_path = base_output_paths.results / "config.yaml"
            OmegaConf.save(config=config_snapshot, f=str(config_path))
            logger.info("📦 Saved final config snapshot to %s", config_path)
        process_graph_generator.save_graph(str(base_output_paths.plots / "process_graph_output.svg"))

        logger.info("🔥 GTSFM: Running image pair retrieval...")
        retriever_metrics, visibility_graph, similarity_matrix = self._run_retriever(client, base_output_paths)
        base_metrics_groups.append(retriever_metrics)
        image_future_map = self.loader.get_image_futures(client)
        one_view_data_dict = self.loader.get_one_view_data_dict()

        # Global two-view verification: run the frontend ONCE over the full retrieval graph and keep only
        # the edges where a two-view model was verified. Everything downstream — partitioning, per-cluster
        # reconstruction, merging — consumes the VERIFIED graph, so clusters are never carved along
        # similarity edges that have no verifiable geometry. Optimizers without a two-view frontend
        # (pure feedforward, e.g. ClusterVGGT / AnySplat) cannot verify and keep the retrieval graph.
        padded_keypoints_list: Optional[list[Keypoints]] = None
        v_corr_idxs_dict: Optional[AnnotatedGraph[np.ndarray]] = None
        correspondence_generator = getattr(self.cluster_optimizer, "correspondence_generator", None)
        two_view_estimator = getattr(self.cluster_optimizer, "two_view_estimator", None)
        if correspondence_generator is not None and two_view_estimator is not None:
            retrieval_edge_count = len(visibility_graph)
            logger.info("🔎 GTSFM: Global two-view verification over %d retrieval edges...", retrieval_edge_count)
            padded_keypoints_list, v_corr_idxs_dict = verify_visibility_graph(
                client,
                correspondence_generator,
                two_view_estimator,
                self.loader,
                image_future_map,
                one_view_data_dict,
                visibility_graph,
            )
            verified_graph = sorted(v_corr_idxs_dict.keys())
            all_nodes = visibility_graph_keys(verified_graph)
            largest_cc = set(get_nodes_in_largest_connected_component(verified_graph)) if verified_graph else set()
            logger.info(
                "🔎 Verified graph: %d/%d edges; nodes=%d, largest_cc=%d, dropped_by_partition=%d",
                len(verified_graph),
                retrieval_edge_count,
                len(all_nodes),
                len(largest_cc),
                len(all_nodes) - len(largest_cc),
            )
            visibility_graph = verified_graph

        # Bridge reconnection: add cross-component edges to reconnect island components.
        if similarity_matrix is not None and self._bridge_min_similarity > 0:
            from gtsfm.utils.viewgraph_reconnector import reconnect_visibility_graph

            bridge_result = reconnect_visibility_graph(
                visibility_graph=visibility_graph,
                similarity_matrix=similarity_matrix,
                min_bridge_similarity=self._bridge_min_similarity,
                top_k_per_component=self._bridge_top_k,
                min_component_size=self._bridge_min_component_size,
            )
            if bridge_result.bridge_edges:
                logger.info(
                    "🌉 Bridge reconnection: added %d edges, components %d -> %d " "(reconnected %d, unreachable %d)",
                    len(bridge_result.bridge_edges),
                    bridge_result.num_components_before,
                    bridge_result.num_components_after,
                    bridge_result.components_reconnected,
                    bridge_result.components_unreachable,
                )
                visibility_graph = bridge_result.reconnected_graph
            del similarity_matrix

        # Graph partitioning: Divide the visibility graph into clusters (runs eagerly, no delayed/futures).
        logger.info("🔥 GTSFM: Partitioning the view graph...")
        assert self.graph_partitioner is not None, "Graph partitioner is not set up!"
        cluster_tree = self.graph_partitioner.run(visibility_graph)
        self.graph_partitioner.log_partition_details(cluster_tree, base_output_paths)
        save_retrieval_two_view_metrics(base_output_paths)

        logger.info("🔥 GTSFM: Scheduling cluster optimizations...")
        merged_scene: Optional[cluster_merging.MergedNodeSummary] = None

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
                        global_v_corr_idxs_dict=v_corr_idxs_dict,
                        global_keypoints=padded_keypoints_list,
                    )

                context_tree = cluster_tree.map_with_path(to_context)

                # Runs reconstruction on each node of the VisibilityGraph (with context) tree.
                # Returns handles to various outputs: reconstruction, metrics, io_barrier etc.
                handles_tree = context_tree.map(self._schedule_single_cluster)

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
                        options=self._merging_options,
                    )

                merged_future_tree = submit_tree_map_with_children(client, reconstruction_tree, merge_fn)
                export_tree = cluster_merging.schedule_exports(client, handles_tree, merged_future_tree)
                summary_tree = cluster_merging.schedule_summaries(client, merged_future_tree)
                root_merge_summary: Optional[cluster_merging.MergedNodeSummary] = None
                for handle_node, summary_node, export_node in zip(
                    PreOrderIter(handles_tree),
                    PreOrderIter(summary_tree),
                    PreOrderIter(export_tree),
                ):
                    handle = handle_node.value
                    summary_future = summary_node.value
                    export_future = export_node.value

                    metrics_groups = list(handle.metrics.result())
                    handle.io_barrier.result()
                    export_future.result()
                    if handle.cluster_path == ():
                        merged_summary = summary_future.result()
                        base_metrics_groups.extend(metrics_groups)
                        base_metrics_groups.append(merged_summary.metrics)
                        base_metrics_groups.append(merged_summary.pre_ba_metrics)
                        root_merge_summary = merged_summary
                    else:
                        merged_summary = summary_future.result()
                        metrics_groups.append(merged_summary.metrics)
                        metrics_groups.append(merged_summary.pre_ba_metrics)
                        save_metrics_reports(metrics_groups, str(handle.output_paths.metrics))
                if root_merge_summary is not None:
                    logger.info("🔥 GTSFM: Running cluster optimization and merging...")
                    merged_scene = root_merge_summary

        if merged_scene is not None and merged_scene.merge_success:
            logger.info(
                "Merged scene contains %d images and %d tracks.",
                merged_scene.num_images,
                merged_scene.num_tracks,
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

    def _run_retriever(
        self, client: Client, output_paths: OutputPaths
    ) -> tuple[GtsfmMetricsGroup, VisibilityGraph, Optional[object]]:
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

        # Grab the similarity matrix BEFORE save_diagnostics clears it (sets to None).
        similarity_matrix = getattr(retriever, "_latest_similarity_matrix", None)
        if similarity_matrix is None:
            # Handle JointSimilaritySequentialRetriever wrapper.
            inner = getattr(retriever, "_similarity_retriever", None)
            if inner is not None:
                similarity_matrix = getattr(inner, "_latest_similarity_matrix", None)

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

        return retriever_metrics, visibility_graph, similarity_matrix
