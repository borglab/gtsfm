"""Base class for runner that executes SfM."""

import argparse
import os
import time
from abc import abstractmethod
from pathlib import Path
import dask
import hydra
from dask.distributed import Client, LocalCluster, SSHCluster, performance_report
from hydra.utils import instantiate
from omegaconf import OmegaConf

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase, ImageMatchingRegime
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator

logger = logger_utils.get_logger()

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent


class GtsfmRunnerBase:
    def __init__(self, tag: str) -> None:
        self._tag: str = tag
        argparser: argparse.ArgumentParser = self.construct_argparser()
        self.parsed_args: argparse.Namespace = argparser.parse_args()
        if self.parsed_args.dask_tmpdir:
            dask.config.set({"temporary_directory": DEFAULT_OUTPUT_ROOT / self.parsed_args.dask_tmpdir})

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
            help="Master config, including back-end configuration. Choose sift_front_end.yaml or deep_front_end.yaml",
        )
        parser.add_argument(
            "--correspondence_generator_config_name",
            type=str,
            default=None,
            help="Override flag for correspondence generator (choose from among gtsfm/configs/correspondence).",
        )
        parser.add_argument(
            "--verifier_config_name",
            type=str,
            default=None,
            help="Override flag for verifier (choose from among gtsfm/configs/verifier).",
        )
        parser.add_argument(
            "--retriever_config_name",
            type=str,
            default=None,
            help="Override flag for retriever (choose from among gtsfm/configs/retriever).",
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
            default=None,
            help="maximum number of consecutive frames to consider for matching/co-visibility",
        )
        parser.add_argument(
            "--share_intrinsics", action="store_true", help="Shares the intrinsics between all the cameras"
        )
        parser.add_argument("--mvs_off", action="store_true", help="Turn off dense MVS reconstruction")
        parser.add_argument(
            "--output_root",
            type=str,
            default=DEFAULT_OUTPUT_ROOT,
            help="Root directory. Results, plots and metrics will be stored in subdirectories,"
            " e.g. {output_root}/results",
        )
        parser.add_argument(
            "--dask_tmpdir",
            type=str,
            default=None,
            help="tmp directory for dask workers, uses dask's default (/tmp) if not set",
        )
        parser.add_argument(
            "--cluster_config",
            type=str,
            default=None,
            help="config listing IP worker addresses for the cluster,"
            " first worker is used as scheduler and should contain the dataset",
        )
        parser.add_argument(
            "--dashboard_port",
            type=str,
            default=":8787",
            help="dask dashboard port number",
        )
        return parser

    @abstractmethod
    def construct_loader(self) -> LoaderBase:
        pass

    def construct_scene_optimizer(self) -> SceneOptimizer:
        """Construct scene optimizer.

        All configs are relative to the gtsfm module.
        """
        with hydra.initialize_config_module(config_module="gtsfm.configs"):
            overrides = ["+SceneOptimizer.output_root=" + str(self.parsed_args.output_root)]
            if self.parsed_args.share_intrinsics:
                overrides.append("SceneOptimizer.multiview_optimizer.bundle_adjustment_module.shared_calib=True")

            main_cfg = hydra.compose(
                config_name=self.parsed_args.config_name,
                overrides=overrides,
            )
            scene_optimizer: SceneOptimizer = instantiate(main_cfg.SceneOptimizer)

        # Override correspondence generator.
        if self.parsed_args.correspondence_generator_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.correspondence"):
                correspondence_cfg = hydra.compose(
                    config_name=self.parsed_args.correspondence_generator_config_name,
                )
                logger.info("\n\nCorrespondenceGenerator override: " + OmegaConf.to_yaml(correspondence_cfg))
                scene_optimizer.correspondence_generator: CorrespondenceGeneratorBase = instantiate(
                    correspondence_cfg.CorrespondenceGenerator
                )

        # Override verifier.
        if self.parsed_args.verifier_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.verifier"):
                verifier_cfg = hydra.compose(
                    config_name=self.parsed_args.verifier_config_name,
                )
                logger.info("\n\nVerifier override: " + OmegaConf.to_yaml(verifier_cfg))
                scene_optimizer.two_view_estimator._verifier: VerifierBase = instantiate(verifier_cfg.verifier)

        # Override retriever.
        if self.parsed_args.retriever_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.retriever"):
                retriever_cfg = hydra.compose(
                    config_name=self.parsed_args.retriever_config_name,
                )
                logger.info("\n\nRetriever override: " + OmegaConf.to_yaml(retriever_cfg))
                scene_optimizer.retriever: RetrieverBase = instantiate(retriever_cfg.retriever)

        if self.parsed_args.max_frame_lookahead is not None:
            if scene_optimizer.retriever._matching_regime in [
                ImageMatchingRegime.SEQUENTIAL,
                ImageMatchingRegime.SEQUENTIAL_HILTI,
            ]:
                scene_optimizer.retriever._max_frame_lookahead = self.parsed_args.max_frame_lookahead
            elif scene_optimizer.retriever._matching_regime == ImageMatchingRegime.SEQUENTIAL_WITH_RETRIEVAL:
                scene_optimizer.retriever._seq_retriever._max_frame_lookahead = self.parsed_args.max_frame_lookahead

        if self.parsed_args.mvs_off:
            scene_optimizer.run_dense_optimizer = False

        logger.info("\n\nSceneOptimizer: " + str(scene_optimizer))
        return scene_optimizer

    def run(self) -> None:
        """Run the SceneOptimizer."""
        start_time = time.time()

        # create dask cluster
        if self.parsed_args.cluster_config:
            workers = OmegaConf.load(
                os.path.join(self.parsed_args.output_root, "gtsfm", "configs", self.parsed_args.cluster_config)
            )["workers"]
            scheduler = workers[0]
            cluster = SSHCluster(
                [scheduler] + workers,
                scheduler_options={"dashboard_address": self.parsed_args.dashboard_port},
                worker_options={
                    "n_workers": self.parsed_args.num_workers,
                    "nthreads": self.parsed_args.threads_per_worker,
                },
            )
            client = Client(cluster)
            # getting first worker's IP address and port to do IO
            io_worker = list(client.scheduler_info()["workers"].keys())[0]
            self.loader._input_worker = io_worker
            self.scene_optimizer._output_worker = io_worker
        else:
            cluster = LocalCluster(
                n_workers=self.parsed_args.num_workers, threads_per_worker=self.parsed_args.threads_per_worker
            )
            client = Client(cluster)

        # create process graph
        process_graph_generator = ProcessGraphGenerator()
        if type(self.scene_optimizer.correspondence_generator) == ImageCorrespondenceGenerator:
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

        pairs_graph = self.scene_optimizer.retriever.create_computation_graph(self.loader)
        with performance_report(filename="retriever-dask-report.html"):
            image_pair_indices = pairs_graph.compute()

        (
            delayed_keypoints,
            delayed_putative_corr_idxs_dict,
        ) = self.scene_optimizer.correspondence_generator.create_computation_graph(
            delayed_images=self.loader.create_computation_graph_for_images(),
            delayed_image_shapes=self.loader.create_computation_graph_for_image_shapes(),
            image_pair_indices=image_pair_indices,
        )

        with performance_report(filename="correspondence-generator-dask-report.html"):
            keypoints_list, putative_corr_idxs_dict = dask.compute(delayed_keypoints, delayed_putative_corr_idxs_dict)

        delayed_sfm_result, delayed_io = self.scene_optimizer.create_computation_graph(
            keypoints_list=keypoints_list,
            putative_corr_idxs_dict=putative_corr_idxs_dict,
            num_images=len(self.loader),
            image_pair_indices=image_pair_indices,
            image_graph=self.loader.create_computation_graph_for_images(),
            all_intrinsics=self.loader.create_computation_graph_for_intrinsics(),
            relative_pose_priors=self.loader.get_relative_pose_priors(image_pair_indices),
            absolute_pose_priors=self.loader.get_absolute_pose_priors(),
            cameras_gt=self.loader.create_computation_graph_for_gt_cameras(),
            gt_wTi_list=self.loader.get_gt_poses(),
        )

        with performance_report(filename="scene-optimizer-dask-report.html"):
            sfm_result, *io = dask.compute(delayed_sfm_result, *delayed_io)

        assert isinstance(sfm_result, GtsfmData)

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)
