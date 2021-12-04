import argparse
import time
from abc import abstractmethod
from typing import Optional

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

        return parser

    @abstractmethod
    def construct_loader(self) -> LoaderBase:
        """Initializes the relevant Loader."""

    def construct_scene_optimizer(self) -> SceneOptimizer:
        """Initializes the SceneOptimizer from input configuratio file using Hydra."""
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

    def scatter_scene_mesh(self, client: Client) -> Optional[GtsfmData]:
        """Scatters ground truth scene mesh into distributed memory to preserve computation time and memory.

        Ref: http://distributed.dask.org/en/stable/api.html#distributed.Client.scatter
        """
        # TODO(travisdriver): find a way to scatter mesh that doesn't require copying the GtsfmData object.
        if self.loader.gt_gtsfm_data.scene_mesh is None:
            return self.loader.gt_gtsfm_data
        gt_gtsfm_data = GtsfmData(
            self.loader.gt_gtsfm_data.number_images,
            self.loader.gt_gtsfm_data._cameras,
            self.loader.gt_gtsfm_data._tracks,
        )
        gt_gtsfm_data.add_scene_mesh(client.scatter(self.loader.gt_gtsfm_data.scene_mesh, broadcast=True))
        logger.info("Scattered scene mesh.")

        return gt_gtsfm_data

    def run(self) -> None:
        """Run Structure-from-Motion (SfM) pipeline."""
        start_time = time.time()

        # Create dask client.
        cluster = LocalCluster(
            n_workers=self.parsed_args.num_workers, threads_per_worker=self.parsed_args.threads_per_worker
        )

        with Client(cluster) as client, performance_report(filename="dask-report.html"):
            # Scatter scene mesh across all nodes to preserve computation time and memory.
            gt_gtsfm_data = self.scatter_scene_mesh(client)

            # Prepare computation graph.
            sfm_result_graph = self.scene_optimizer.create_computation_graph(
                num_images=len(self.loader),
                image_pair_indices=self.loader.get_valid_pairs(),
                image_graph=self.loader.create_computation_graph_for_images(),
                camera_intrinsics_graph=self.loader.create_computation_graph_for_intrinsics(),
                image_shape_graph=self.loader.create_computation_graph_for_image_shapes(),
                gt_gtsfm_data=gt_gtsfm_data,
            )

            # Run SfM pipeline.
            sfm_result = sfm_result_graph.compute()

        assert isinstance(sfm_result, GtsfmData)
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", (time.time() - start_time) / 60)
