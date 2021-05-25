import argparse

import hydra
import numpy as np
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.scene_optimizer import SceneOptimizer

from gtsfm.loader.argoverse_dataset_loader import ArgoverseDatasetLoader
from gtsfm.utils.logger import get_logger

logger = get_logger()


def run_scene_optimizer(args) -> None:
    """ Run GTSFM over images from an Argoverse vehicle log"""
    with hydra.initialize_config_module(config_module="gtsfm.configs"):
        # config is relative to the gtsfm module
        cfg = hydra.compose(config_name="default_lund_door_set1_config.yaml")
        scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

        loader = ArgoverseDatasetLoader(
            dataset_dir=args.dataset_dir,
            log_id=args.log_id,
            stride=args.stride,
            max_num_imgs=args.max_num_imgs,
            max_lookahead_sec=args.max_lookahead_sec,
            camera_name=args.camera_name,
        )

        sfm_result_graph = scene_optimizer.create_computation_graph(
            len(loader),
            loader.get_valid_pairs(),
            loader.create_computation_graph_for_images(),
            loader.create_computation_graph_for_intrinsics(),
            use_intrinsics_in_verification=True,
            gt_pose_graph=loader.create_computation_graph_for_poses(),
        )

        # create dask client
        cluster = LocalCluster(n_workers=2, threads_per_worker=4)

        with Client(cluster), performance_report(filename="dask-report.html"):
            sfm_result = sfm_result_graph.compute()

        assert isinstance(sfm_result, GtsfmData)
        scene_avg_reproj_error = sfm_result.get_scene_avg_reprojection_error()
        logger.info('Scene avg reproj error: {}'.format(str(np.round(scene_avg_reproj_error, 3))))


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
    args = parser.parse_args()
    logger.info(args)

    run_scene_optimizer(args)
