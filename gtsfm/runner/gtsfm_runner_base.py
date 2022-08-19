"""Base class for runner that executes SfM."""

import argparse
import time
from abc import abstractmethod
from pathlib import Path

import dask
import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate
from omegaconf import OmegaConf

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.exhaustive_retriever import ExhaustiveRetriever
from gtsfm.retriever.joint_netvlad_sequential_retriever import JointNetVLADSequentialRetriever
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever
from gtsfm.retriever.retriever_base import ImageMatchingRegime
from gtsfm.retriever.rig_retriever import RigRetriever
from gtsfm.retriever.sequential_hilti_retriever import SequentialHiltiRetriever
from gtsfm.retriever.sequential_retriever import SequentialRetriever
from gtsfm.scene_optimizer import SceneOptimizer

logger = logger_utils.get_logger()

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent


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
            choices=[
                "exhaustive",
                "sequential",
                "sequential_hilti",
                "rig_hilti",
                "retrieval",
                "sequential_with_retrieval",
            ],
            default="sequential",
            help="Choose mode for matching.",
        )
        parser.add_argument(
            "--num_matched",
            type=int,
            default=5,
            help="Number of matches to provide per retrieval query.",
        )
        parser.add_argument(
            "--share_intrinsics", action="store_true", help="Shares the intrinsics between all the cameras"
        )
        parser.add_argument(
            "--output_root",
            type=str,
            default=DEFAULT_OUTPUT_ROOT,
            help="Root directory. Results, plots and metrics will be stored in subdirectories,"
            "e.g. {output_root}/results",
        )
        return parser

    @abstractmethod
    def construct_loader(self) -> LoaderBase:
        pass

    def construct_scene_optimizer(self) -> SceneOptimizer:
        """Construct scene optimizer."""
        with hydra.initialize_config_module(config_module="gtsfm.configs"):
            overrides = ["+SceneOptimizer.output_root=" + str(self.parsed_args.output_root)]
            if self.parsed_args.share_intrinsics:
                overrides.append("SceneOptimizer.multiview_optimizer.bundle_adjustment_module.shared_calib=True")

            # config is relative to the gtsfm module
            cfg = hydra.compose(
                config_name=self.parsed_args.config_name,
                overrides=overrides,
            )
            logger.info("Using config: ")
            logger.info(OmegaConf.to_yaml(cfg))
            scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

        return scene_optimizer

    def construct_retriever(self):
        """Set up retriever module."""
        matching_regime = ImageMatchingRegime(self.parsed_args.matching_regime)

        if matching_regime == ImageMatchingRegime.EXHAUSTIVE:
            retriever = ExhaustiveRetriever()

        elif matching_regime == ImageMatchingRegime.RETRIEVAL:
            retriever = NetVLADRetriever(num_matched=self.parsed_args.num_matched)

        elif matching_regime == ImageMatchingRegime.SEQUENTIAL:
            retriever = SequentialRetriever(max_frame_lookahead=self.parsed_args.max_frame_lookahead)

        elif matching_regime == ImageMatchingRegime.SEQUENTIAL_WITH_RETRIEVAL:
            retriever = JointNetVLADSequentialRetriever(
                num_matched=self.parsed_args.num_matched, max_frame_lookahead=self.parsed_args.max_frame_lookahead
            )

        elif matching_regime == ImageMatchingRegime.SEQUENTIAL_HILTI:
            retriever = SequentialHiltiRetriever(max_frame_lookahead=self.parsed_args.max_frame_lookahead)

        elif matching_regime == ImageMatchingRegime.RIG_HILTI:
            retriever = RigRetriever(threshold=self.parsed_args.proxy_threshold)

        return retriever

    def run(self) -> None:
        """Run the SceneOptimizer."""
        start_time = time.time()

        # create dask client
        cluster = LocalCluster(
            n_workers=self.parsed_args.num_workers, threads_per_worker=self.parsed_args.threads_per_worker
        )

        pairs_graph = self.retriever.create_computation_graph(self.loader)
        with Client(cluster), performance_report(filename="dask-report.html"):
            image_pair_indices = pairs_graph.compute()

        delayed_sfm_result, delayed_io = self.scene_optimizer.create_computation_graph(
            num_images=len(self.loader),
            image_pair_indices=image_pair_indices,
            image_graph=self.loader.create_computation_graph_for_images(),
            all_intrinsics=self.loader.get_all_intrinsics(),
            image_shapes=self.loader.get_image_shapes(),
            relative_pose_priors=self.loader.get_relative_pose_priors(image_pair_indices),
            absolute_pose_priors=self.loader.get_absolute_pose_priors(),
            cameras_gt=self.loader.get_gt_cameras(),
            gt_wTi_list=self.loader.get_gt_poses(),
            matching_regime=ImageMatchingRegime(self.parsed_args.matching_regime),
        )

        with Client(cluster), performance_report(filename="dask-report.html"):
            sfm_result, *io = dask.compute(delayed_sfm_result, *delayed_io)

        assert isinstance(sfm_result, GtsfmData)

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)
