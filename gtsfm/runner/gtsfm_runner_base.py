import argparse
import time
from abc import abstractmethod

import dask
import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate
from omegaconf import OmegaConf

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.scene_optimizer import SceneOptimizer

from gtsfm.retriever.exhaustive_retriever import ExhaustiveRetriever
from gtsfm.retriever.retriever_base import ImageMatchingRegime
from gtsfm.retriever.rig_retriever import RigRetriever
from gtsfm.retriever.sequential_hilti_retriever import SequentialHiltiRetriever
from gtsfm.retriever.sequential_retriever import SequentialRetriever


logger = logger_utils.get_logger()


class GtsfmRunnerBase:
    def __init__(self, tag: str):
        self._tag: str = tag
        argparser: argparse.ArgumentParser = self.construct_argparser()
        self.parsed_args: argparse.Namespace = argparser.parse_args()

        self.loader: LoaderBase = self.construct_loader()
        self.retriever = self.construct_retriever()
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
            "--matching_regime",
            type=str,
            choices=["exhaustive", "sequential", "sequential_hilti", "rig_hilti"],
            default="sequential",
            help="Choose mode for matching.",
        )
        parser.add_argument(
            "--share_intrinsics", action="store_true", help="Shares the intrinsics between all the cameras"
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
                overrides=["SceneOptimizer.multiview_optimizer.bundle_adjustment_module.shared_calib=True"]
                if self.parsed_args.share_intrinsics
                else [],
            )
            logger.info("[runner] Using config: ")
            logger.info(OmegaConf.to_yaml(cfg))
            scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

        return scene_optimizer

    def construct_retriever(self):
        """Set up retriever module."""
        matching_regime = ImageMatchingRegime(self.parsed_args.matching_regime)

        if matching_regime == ImageMatchingRegime.EXHAUSTIVE:
            retriever = ExhaustiveRetriever()

        elif matching_regime == ImageMatchingRegime.SEQUENTIAL:
            retriever = SequentialRetriever(max_frame_lookahead=self.parsed_args.max_frame_lookahead)

        elif matching_regime == ImageMatchingRegime.SEQUENTIAL_HILTI:
            retriever = SequentialHiltiRetriever(max_frame_lookahead=self.parsed_args.max_frame_lookahead)

        elif matching_regime == ImageMatchingRegime.RIG_HILTI:
            retriever = RigRetriever(threshold=self.parsed_args.proxy_threshold, subsample=self.parsed_args.subsample)

        return retriever

    def run(self) -> None:
        """Run the SceneOptimizer."""
        start_time = time.time()

        # create dask client
        cluster = LocalCluster(
            n_workers=self.parsed_args.num_workers,
            threads_per_worker=self.parsed_args.threads_per_worker,
            memory_limit="8GB",
        )

        pairs_graph = self.retriever.create_computation_graph(self.loader)
        with Client(cluster), performance_report(filename="dask-report.html"):
            image_pair_indices = pairs_graph.compute()

        # Compute which images are actually involved in visual matching.
        images_involved = set()
        for i1, i2 in image_pair_indices:
            images_involved.add(i1)
            images_involved.add(i2)
        logger.info(f"[runner] There are {len(images_involved)} images involved.")
        all_delayed_images = self.loader.create_computation_graph_for_images()
        delayed_images = {i: all_delayed_images[i] for i in images_involved}

        delayed_sfm_result, delayed_io = self.scene_optimizer.create_computation_graph(
            num_images=len(self.loader),
            image_pair_indices=image_pair_indices,
            delayed_images=delayed_images,
            all_intrinsics=self.loader.get_all_intrinsics(),
            image_shapes=self.loader.get_image_shapes(),
            relative_pose_priors=self.loader.get_relative_pose_priors(),
            absolute_pose_priors=self.loader.get_absolute_pose_priors(),
            cameras_gt=self.loader.get_gt_cameras(),
            gt_wTi_list=self.loader.get_gt_poses(),
            image_fnames=self.loader.get_image_fnames(),
            matching_regime=ImageMatchingRegime(self.parsed_args.matching_regime),
        )

        with Client(cluster), performance_report(filename="dask-report.html"):
            sfm_result, *io = dask.compute(delayed_sfm_result, *delayed_io)

        assert isinstance(sfm_result, GtsfmData)

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("[runner] GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)
