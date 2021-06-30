import argparse
from pathlib import Path

import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.scene_optimizer import SceneOptimizer

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"

logger = logger_utils.get_logger()


def run_scene_optimizer(args) -> None:
    """ """
    with hydra.initialize_config_module(config_module="gtsfm.configs"):
        # config is relative to the gtsfm module
        cfg = hydra.compose(config_name=args.config_name)

        scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

        loader = ColmapLoader(
            colmap_files_dirpath=args.colmap_files_dirpath,
            images_dir=args.images_dir,
            max_frame_lookahead=args.max_frame_lookahead,
        )

        sfm_result_graph = scene_optimizer.create_computation_graph(
            num_images=len(loader),
            image_pair_indices=loader.get_valid_pairs(),
            image_graph=loader.create_computation_graph_for_images(),
            camera_intrinsics_graph=loader.create_computation_graph_for_intrinsics(),
            image_shape_graph=loader.create_computation_graph_for_image_shapes(),
            gt_pose_graph=loader.create_computation_graph_for_poses(),
        )

        # create dask client
        cluster = LocalCluster(n_workers=args.num_workers, threads_per_worker=args.threads_per_worker)

        with Client(cluster), performance_report(filename="dask-report.html"):
            sfm_result = sfm_result_graph.compute()

        assert isinstance(sfm_result, GtsfmData)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GTSFM with intrinsics and image names stored in COLMAP-format")
    parser.add_argument(
        "--images_dir", type=str, required=True, help="path to directory containing png, jpeg, or jpg images files"
    )
    parser.add_argument(
        "--colmap_files_dirpath",
        type=str,
        required=True,
        help="path to directory containing images.txt, points3D.txt, and cameras.txt",
    )
    parser.add_argument(
        "--max_frame_lookahead",
        type=int,
        default=1,
        help="maximum number of consecutive frames to consider for matching/co-visibility",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to start (processes, by default)",
    )
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=1,
        help="Number of threads per each worker",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="deep_front_end.yaml",
        help="Choose sift_front_end.yaml or deep_front_end.yaml",
    )

    args = parser.parse_args()

    run_scene_optimizer(args)
