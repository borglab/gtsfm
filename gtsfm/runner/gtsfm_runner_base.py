import argparse
import time
from abc import abstractmethod

import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate
from omegaconf import OmegaConf

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.scene_optimizer import SceneOptimizer


logger = logger_utils.get_logger()


class GtsfmRunnerBase:
    def __init__(self, tag: str):
        self._tag: str = tag
        argparser: argparse.ArgumentParser = self.construct_argparser()
        self.parsed_args: argparse.Namespace = argparser.parse_args()

        self.loader: LoaderBase = self.construct_loader()
        self.scene_optimizer: SceneOptimizer = self.construct_scene_optimizer()

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self._tag)

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

        parser.add_argument(
            "--max_frame_lookahead",
            type=int,
            default=20,
            help="maximum number of consecutive frames to consider for matching/co-visibility",
        )

        parser.add_argument(
            "--share_intrinsics", action="store_true", help="Shares the intrinsics between all the cameras"
        )

        parser.add_argument(
            "--max_iters",
            type=int,
            default=10,
            help="max number ransac iters",
        )

        parser.add_argument(
            "--success_prob",
            type=float,
            default=0.9,
            help="RANSAC guaranteed success probability",
        )

        return parser

    @abstractmethod
    def construct_loader(self) -> LoaderBase:
        pass

    def construct_scene_optimizer(self) -> SceneOptimizer:
        with hydra.initialize_config_module(config_module="gtsfm.configs"):
            # config is relative to the gtsfm module
            cfg = hydra.compose(
                config_name=self.parsed_args.config_name,
                overrides=[
                    f"SceneOptimizer.two_view_estimator.verifier.success_prob={self.parsed_args.success_prob}",
                    f"SceneOptimizer.two_view_estimator.verifier.max_iters={self.parsed_args.max_iters}"
                ]
            )
            logger.info("Using config: ")
            logger.info(OmegaConf.to_yaml(cfg))
            scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

        return scene_optimizer

    def run(self) -> None:
        start_time = time.time()
        sfm_result_graph = self.scene_optimizer.create_computation_graph(
            num_images=len(self.loader),
            image_pair_indices=self.loader.get_valid_pairs(),
            image_graph=self.loader.create_computation_graph_for_images(),
            camera_intrinsics_graph=self.loader.create_computation_graph_for_intrinsics(),
            image_shape_graph=self.loader.create_computation_graph_for_image_shapes(),
            gt_cameras_graph=self.loader.create_computation_graph_for_cameras(),
        )

        # create dask client
        cluster = LocalCluster(
            n_workers=self.parsed_args.num_workers, threads_per_worker=self.parsed_args.threads_per_worker
        )

        with Client(cluster), performance_report(filename="dask-report.html"):
            sfm_result = sfm_result_graph.compute()

        assert isinstance(sfm_result, GtsfmData)

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)
