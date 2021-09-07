"""Runs GTSfM on an AstroNet dataset.

Author: Travis Driver
"""

import argparse
from pathlib import Path

import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.astronet_loader import AstroNetLoader
from gtsfm.scene_optimizer import SceneOptimizer

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"

logger = logger_utils.get_logger()


def run_scene_optimizer() -> None:
    """Runs SceneOptimizer on AstroNet data."""
    # Parse inputs.
    parser = argparse.ArgumentParser(description="Run GTSfM on AstroNet segment.")
    parser.add_argument(
        "--data_dir", "-d", type=str, required=True, help="path to directory containing AstroNet segment"
    )
    parser.add_argument(
        "--max_frame_lookahead",
        "-l",
        type=int,
        default=5,
        help="maximum number of consecutive frames to consider for matching/co-visibility",
    )
    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        default=1,
        help="Number of workers to start (processes, by default)",
    )
    parser.add_argument(
        "--threads_per_worker",
        "-th",
        type=int,
        default=1,
        help="Number of threads per each worker",
    )
    parser.add_argument(
        "--config_name",
        "-c",
        type=str,
        default="deep_front_end.yaml",
        help="Choose sift_front_end.yaml or deep_front_end.yaml",
    )
    parser.add_argument(
        "--scene_mesh_path",
        "-m",
        type=str,
        default=None,
        help="Path to file containing triangular surface mesh of target body.",
    )
    args = parser.parse_args()

    # Initialize SceneOptimizer from config.
    with hydra.initialize_config_module(config_module="gtsfm.configs"):  # config is relative to the gtsfm module
        cfg = hydra.compose(config_name=args.config_name)
    scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

    # Initialize loader.
    loader = AstroNetLoader(
        data_dir=args.data_dir,
        use_gt_extrinsics=True,
        use_gt_sfmtracks=False,
        max_frame_lookahead=args.max_frame_lookahead,
    )

    # Create Dask task graph.
    # Note: scene mesh not surrently used by scene_optimizer
    sfm_result_graph = scene_optimizer.create_computation_graph(
        num_images=len(loader),
        image_pair_indices=loader.get_valid_pairs(),
        image_graph=loader.create_computation_graph_for_images(),
        camera_intrinsics_graph=loader.create_computation_graph_for_intrinsics(),
        image_shape_graph=loader.create_computation_graph_for_image_shapes(),
        gt_pose_graph=loader.create_computation_graph_for_poses(),
    )

    # Create Dask client.
    cluster = LocalCluster(n_workers=args.num_workers, threads_per_worker=args.threads_per_worker)

    # Compute SfM solution.
    with Client(cluster), performance_report(filename="dask-report.html"):
        sfm_result = sfm_result_graph.compute()
    assert isinstance(sfm_result, GtsfmData)


if __name__ == "__main__":
    run_scene_optimizer()
