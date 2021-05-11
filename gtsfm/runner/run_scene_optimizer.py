
import argparse
import os
from pathlib import Path

from dask.distributed import Client, LocalCluster, performance_report
from hydra.experimental import compose, initialize_config_module
from hydra.utils import instantiate

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.scene_optimizer import SceneOptimizer

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"

logger = logger_utils.get_logger()


def run_scene_optimizer(args) -> None:
    """ """
    with initialize_config_module(config_module="gtsfm.configs"):
        # config is relative to the gtsfm module
        cfg = compose(config_name="default_lund_door_set1_config.yaml")
        scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

        loader = OlssonLoader(args.dataset_root, image_extension=args.image_extension)

        sfm_result_graph = scene_optimizer.create_computation_graph(
            len(loader),
            loader.get_valid_pairs(),
            loader.create_computation_graph_for_images(),
            loader.create_computation_graph_for_intrinsics(),
            gt_pose_graph=loader.create_computation_graph_for_poses(),
        )

        # create dask client
        cluster = LocalCluster(n_workers=2, threads_per_worker=4)

        with Client(cluster), performance_report(filename="dask-report.html"):
            sfm_result = sfm_result_graph.compute()

        assert isinstance(sfm_result, GtsfmData)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GTSFM with intrinsics and image names stored in COLMAP-format")
    parser.add_argument(
        "--dataset_root", type=str, default=os.path.join(DATA_ROOT, "set1_lund_door"), help=""
    )
    parser.add_argument(
        "--image_extension", type=str, default="JPG", help=""
    )
    args = parser.parse_args()

    run_scene_optimizer(args)
