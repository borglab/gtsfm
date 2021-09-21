import argparse
import os
from pathlib import Path

import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.scene_optimizer import SceneOptimizer

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"

logger = logger_utils.get_logger()


def run_scene_optimizer(args: argparse.Namespace) -> None:
    """ """
    with hydra.initialize_config_module(config_module="gtsfm.configs"):
        # config is relative to the gtsfm module
        cfg = hydra.compose(config_name=args.config_name)
        scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

        loader = OlssonLoader(
            args.dataset_root,
            image_extension=args.image_extension,
            max_frame_lookahead=args.max_frame_lookahead,
            max_resolution=args.max_resolution,
        )

        sfm_result_graph = scene_optimizer.create_computation_graph(
            num_images=len(loader),
            image_pair_indices=loader.get_valid_pairs(),
            image_graph=loader.create_computation_graph_for_images(),
            camera_intrinsics_graph=loader.create_computation_graph_for_intrinsics(),
            image_shape_graph=loader.create_computation_graph_for_image_shapes(),
            gt_cameras_graph=loader.create_computation_graph_for_cameras(),
        )

        # create dask client
        cluster = LocalCluster(n_workers=args.num_workers, threads_per_worker=args.threads_per_worker)

        with Client(cluster), performance_report(filename="dask-report.html"):
            sfm_result = sfm_result_graph.compute()

        assert isinstance(sfm_result, GtsfmData)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTSFM with intrinsics and image names stored in COLMAP-format")
    parser.add_argument("--dataset_root", type=str, default=os.path.join(DATA_ROOT, "set1_lund_door"), help="")
    parser.add_argument("--image_extension", type=str, default="JPG", help="")
    parser.add_argument(
        "--max_frame_lookahead",
        type=int,
        default=20,
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
        default="sift_front_end.yaml",
        help="Choose sift_front_end.yaml or deep_front_end.yaml",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=760,
        help="integer representing maximum length of image's short side"
        " e.g. for 1080p (1920 x 1080), max_resolution would be 1080",
    )
    args = parser.parse_args()

    run_scene_optimizer(args)
