import argparse

import dask
import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.argoverse_dataset_loader import ArgoverseDatasetLoader
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.utils.logger import get_logger

logger = get_logger()


def run_scene_optimizer(args: argparse.Namespace) -> None:
    """Run GTSFM over images from an Argoverse vehicle log"""
    with hydra.initialize_config_module(config_module="gtsfm.configs"):
        # config is relative to the gtsfm module
        cfg = hydra.compose(config_name=args.config_name)
        scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

        loader = ArgoverseDatasetLoader(
            dataset_dir=args.dataset_dir,
            log_id=args.log_id,
            stride=args.stride,
            max_num_imgs=args.max_num_imgs,
            max_lookahead_sec=args.max_lookahead_sec,
            camera_name=args.camera_name,
            max_resolution=args.max_resolution,
        )

        delayed_sfm_result, delayed_io = scene_optimizer.create_computation_graph(
            num_images=len(loader),
            image_pair_indices=loader.get_valid_pairs(),
            image_graph=loader.create_computation_graph_for_images(),
            all_intrinsics=loader.get_computation_graph_for_intrinsics(),
            image_shapes=loader.create_computation_graph_for_image_shapes(),
            cameras_gt=loader.create_computation_graph_for_gt_cameras(),
        )

        # create dask client
        cluster = LocalCluster(n_workers=args.num_workers, threads_per_worker=args.threads_per_worker)

        with Client(cluster), performance_report(filename="dask-report.html"):
            sfm_result, *io = dask.compute(delayed_sfm_result, *delayed_io)

        assert isinstance(sfm_result, GtsfmData)
        scene_avg_reproj_error = sfm_result.get_avg_scene_reprojection_error()
        logger.info("Scene avg reproj error: %.3f", scene_avg_reproj_error)


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_id",
        default="273c1883-673a-36bf-b124-88311b1a80be",
        type=str,
        help="unique ID of Argoverse vehicle log",
    )
    parser.add_argument(
        "--dataset_dir",
        default="/srv/share/cliu324/argoverse-tracking-readonly/train1",
        type=str,
        help="directory where raw Argoverse logs are stored on disk",
    )
    parser.add_argument(
        "--camera_name",
        default="ring_front_center",
        type=str,
        help="Which of 9 Argoverse cameras",
    )
    parser.add_argument(
        "--stride",
        default=10,
        type=int,
        help="image subsampling interval, e.g. every 2 images, every 4 images, etc.",
    )
    parser.add_argument(
        "--max_num_imgs",
        default=20,
        type=int,
        help="maximum number of images to include in dataset (starting from beginning of log sequence)",
    )
    parser.add_argument(
        "--max_lookahead_sec",
        default=2,
        type=float,
        help="",
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
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=760,
        help="integer representing maximum length of image's short side"
        " e.g. for 1080p (1920 x 1080), max_resolution would be 1080",
    )
    args = parser.parse_args()
    logger.info(args)

    run_scene_optimizer(args)
