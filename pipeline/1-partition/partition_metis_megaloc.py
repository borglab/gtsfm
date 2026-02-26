"""Run MegaLoc retrieval + METIS partitioning and persist the visibility graph + cluster tree.

This script mirrors the image_pairs_generator and graph_partitioner configuration in
`gtsfm/configs/vggt.yaml`. It loads images, generates a visibility graph, partitions
the graph with METIS, and saves both the graph and tree under the chosen output root.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import hydra
from dask.distributed import Client, LocalCluster
from hydra.utils import instantiate

import gtsfm.utils.logger as logger_utils
from gtsfm.common.outputs import OutputPaths, prepare_output_paths
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.image_pairs_generator import ImagePairsGenerator

logger = logger_utils.get_logger()


def _build_components(
    config_name: str,
    dataset_dir: str,
    images_dir: str | None,
    max_resolution: int | None,
) -> tuple[LoaderBase, ImagePairsGenerator, GraphPartitionerBase]:
    overrides: list[str] = [f"loader.dataset_dir={dataset_dir}"]
    if images_dir is not None:
        overrides.append(f"loader.images_dir={images_dir}")
    if max_resolution is not None:
        overrides.append(f"loader.max_resolution={max_resolution}")

    with hydra.initialize_config_module(config_module="gtsfm.configs", version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

    loader: LoaderBase = instantiate(cfg.loader)
    image_pairs_generator: ImagePairsGenerator = instantiate(cfg.image_pairs_generator)
    graph_partitioner: GraphPartitionerBase = instantiate(cfg.graph_partitioner)
    return loader, image_pairs_generator, graph_partitioner


def _run_retriever(
    client: Client, loader: LoaderBase, image_pairs_generator: ImagePairsGenerator, output_paths: OutputPaths
) -> VisibilityGraph:
    start_time = time.time()
    batch_size = image_pairs_generator._batch_size
    transforms = image_pairs_generator.get_preprocessing_transforms()
    image_batch_futures = loader.get_all_descriptor_image_batches_as_futures(client, batch_size, *transforms)
    image_fnames = loader.image_filenames()

    logger.info("🔥 Running image pair retrieval...")
    visibility_graph = image_pairs_generator.run(
        client=client,
        image_batch_futures=image_batch_futures,
        image_fnames=image_fnames,
        plots_output_dir=output_paths.plots,
    )

    try:
        image_pairs_generator._retriever.save_diagnostics(
            image_fnames=image_fnames,
            pairs=visibility_graph,
            plots_output_dir=output_paths.plots,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path best-effort
        logger.warning("Failed to persist retriever diagnostics: %s", exc)

    logger.info("🚀 Image pair retrieval took %.2f min.", (time.time() - start_time) / 60.0)
    return visibility_graph


def _save_visibility_graph(graph: VisibilityGraph, output_paths: OutputPaths) -> None:
    try:
        with open(output_paths.results / "visibility_graph.pkl", "wb") as f:
            pickle.dump(graph, f)
    except Exception as exc:
        logger.warning("Failed to serialize visibility graph: %s", exc)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MegaLoc+METIS partitioning and save outputs.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset root containing images/ (Olsson-style loader default).",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Optional path to images directory (overrides loader default).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(Path.cwd()),
        help="Root directory to store results (will create output_root/results).",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="vggt",
        help="Config in gtsfm/configs to load (default: vggt).",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=None,
        help="Override loader max resolution (if unset, uses config default).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of local Dask workers.",
    )
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=1,
        help="Threads per Dask worker.",
    )
    parser.add_argument(
        "--worker_memory_limit",
        type=str,
        default="32GB",
        help="Memory limit per worker, e.g. 16GB.",
    )
    parser.add_argument(
        "--dashboard_address",
        type=str,
        default=":8787",
        help="Dask dashboard address, set to empty string to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_root = Path(args.output_root)
    output_paths = prepare_output_paths(output_root, None)

    loader, image_pairs_generator, graph_partitioner = _build_components(
        config_name=args.config_name,
        dataset_dir=args.dataset_dir,
        images_dir=args.images_dir,
        max_resolution=args.max_resolution,
    )

    logger.info("🌟 Starting Dask local cluster...")
    cluster = LocalCluster(
        n_workers=args.num_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.worker_memory_limit,
        dashboard_address=args.dashboard_address,
    )

    with Client(cluster) as client:
        visibility_graph = _run_retriever(client, loader, image_pairs_generator, output_paths)

    logger.info("🔥 Running METIS partitioning...")
    cluster_tree = graph_partitioner.run(visibility_graph)
    graph_partitioner.log_partition_details(cluster_tree, output_paths)
    _save_visibility_graph(visibility_graph, output_paths)

    logger.info("✅ Saved visibility_graph.pkl and cluster_tree.pkl under %s", output_paths.results)


if __name__ == "__main__":
    main()
