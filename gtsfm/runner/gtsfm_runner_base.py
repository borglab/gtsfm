"""Base class for all runners.

Authors: Ayush Baid
"""
import time
from abc import abstractmethod
from argparse import ArgumentParser, Namespace

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
    """Base class for all runners, which handles argument parsing, loader and SceneOptimizer instantiation and
    execution."""

    def __init__(self, tag: str):
        """Initialize the argument parser, and parses the command line args to create loader and scene optimizer
        objects.

        Args:
            tag: the description associated with the runner, which will be used in the argument parser.
        """
        self._tag: str = tag
        argparser: ArgumentParser = self.construct_argparser()
        self.parsed_args: Namespace = argparser.parse_args()

        self.loader: LoaderBase = self.construct_loader()
        self.scene_optimizer: SceneOptimizer = self.construct_scene_optimizer()

    def construct_argparser(self) -> ArgumentParser:
        """Constructs the argument parser, with the arguments common to all runners. All implementations of the function
        in the child classes of GtsfmRunnerBase should call this method.

        The following command line args are set up in this method:
        - num_workers
        - threads_per_worker
        - config_name
        - max_resolution
        - max_frame_lookahead
        - share_intrinsics

        Returns:
            The argument parser.
        """
        parser = ArgumentParser(description=self._tag)

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

        return parser

    @abstractmethod
    def construct_loader(self) -> LoaderBase:
        """Constructs the loader."""

    def construct_scene_optimizer(self) -> SceneOptimizer:
        """Constructs the scene optimizer."""
        with hydra.initialize_config_module(config_module="gtsfm.configs"):
            # config is relative to the gtsfm module
            cfg = hydra.compose(
                config_name=self.parsed_args.config_name,
                overrides=["SceneOptimizer.multiview_optimizer.bundle_adjustment_module.shared_calib=True"]
                if self.parsed_args.share_intrinsics
                else [],
            )
            logger.info("Using config: ")
            logger.info(OmegaConf.to_yaml(cfg))
            scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

        return scene_optimizer

    def run(self) -> None:
        """Runs GTSFM using the instantiated Loader and SceneOptimizer."""
        start_time = time.time()
        sfm_result_graph = self.scene_optimizer.create_computation_graph(
            num_images=len(self.loader),
            image_pair_indices=self.loader.get_valid_pairs(),
            image_graph=self.loader.create_computation_graph_for_images(),
            camera_intrinsics_graph=self.loader.create_computation_graph_for_intrinsics(),
            image_shape_graph=self.loader.create_computation_graph_for_image_shapes(),
            gt_pose_graph=self.loader.create_computation_graph_for_poses(),
        )

        cluster = LocalCluster(
            n_workers=self.parsed_args.num_workers, threads_per_worker=self.parsed_args.threads_per_worker
        )
        with Client(cluster), performance_report(filename="dask-report.html"):
            sfm_result = sfm_result_graph.compute()

        assert isinstance(sfm_result, GtsfmData)

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)
