"""Base class for runner that executes SfM."""

import argparse
import os
import time
from abc import abstractmethod, abstractproperty
from pathlib import Path
from typing import Any, Dict, Tuple

import dask
import hydra
import numpy as np
from dask.distributed import Client, LocalCluster, SSHCluster, performance_report
from gtsam import Rot3, Unit3
from hydra.utils import instantiate
from omegaconf import OmegaConf

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import ImageMatchingRegime
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.two_view_estimator import TWO_VIEW_OUTPUT, TwoViewEstimationReport, run_two_view_estimator_as_futures
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator

logger = logger_utils.get_logger()

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent


class GtsfmRunnerBase:
    @abstractproperty
    def tag(self):
        pass

    def __init__(self, override_args: Any = None) -> None:
        argparser: argparse.ArgumentParser = self.construct_argparser()
        self.parsed_args: argparse.Namespace = argparser.parse_args(args=override_args)
        if self.parsed_args.dask_tmpdir:
            dask.config.set({"temporary_directory": DEFAULT_OUTPUT_ROOT / self.parsed_args.dask_tmpdir})

        self.loader: LoaderBase = self.construct_loader()
        self.scene_optimizer: SceneOptimizer = self.construct_scene_optimizer()

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.tag)

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
        parser.add_argument(
            "--num_retry_cluster_connection",
            type=int,
            default=3,
            help="number of times to retry cluster connection if it fails",
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
                scene_optimizer.correspondence_generator = instantiate(correspondence_cfg.CorrespondenceGenerator)

        # Override verifier.
        if self.parsed_args.verifier_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.verifier"):
                verifier_cfg = hydra.compose(
                    config_name=self.parsed_args.verifier_config_name,
                )
                logger.info("\n\nVerifier override: " + OmegaConf.to_yaml(verifier_cfg))
                scene_optimizer.two_view_estimator._verifier = instantiate(verifier_cfg.verifier)

        # Override retriever.
        if self.parsed_args.retriever_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.retriever"):
                retriever_cfg = hydra.compose(
                    config_name=self.parsed_args.retriever_config_name,
                )
                logger.info("\n\nRetriever override: " + OmegaConf.to_yaml(retriever_cfg))
                scene_optimizer.retriever = instantiate(retriever_cfg.retriever)

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

        assert False
        logger.info("\n\nSceneOptimizer: " + str(scene_optimizer))
        return scene_optimizer

    def setup_ssh_cluster_with_retries(self):
        """Sets up SSH Cluster allowing multiple retries upon connection failures."""
        workers = OmegaConf.load(
            os.path.join(self.parsed_args.output_root, "gtsfm", "configs", self.parsed_args.cluster_config)
        )["workers"]
        scheduler = workers[0]
        connected = False
        retry_count = 0
        while retry_count < self.parsed_args.num_retry_cluster_connection and not connected:
            logger.info(f"Connecting to the cluster: attempt {retry_count + 1}")
            logger.info(f"Using {scheduler} as scheduler")
            logger.info(f"Using {workers} as workers")
            try:
                cluster = SSHCluster(
                    [scheduler] + workers,
                    scheduler_options={"dashboard_address": self.parsed_args.dashboard_port},
                    worker_options={
                        "n_workers": self.parsed_args.num_workers,
                        "nthreads": self.parsed_args.threads_per_worker,
                    },
                )
                connected = True
            except Exception as e:
                logger.info(f"Worker failed to start: {str(e)}")
                retry_count += 1
        return cluster

    def run(self) -> GtsfmData:
        """Run the SceneOptimizer."""
        assert False
        start_time = time.time()

        # create dask cluster
        if self.parsed_args.cluster_config:
            cluster = self.setup_ssh_cluster_with_retries()
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
        if isinstance(self.scene_optimizer.correspondence_generator, ImageCorrespondenceGenerator):
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

        # TODO(Ayush): Use futures
        image_pair_indices = self.scene_optimizer.retriever.get_image_pairs(
            self.loader, plots_output_dir=self.scene_optimizer._plot_base_path
        )

        intrinsics = self.loader.get_all_intrinsics()

        with performance_report(filename="correspondence-generator-dask-report.html"):
            (
                keypoints_list,
                putative_corr_idxs_dict,
            ) = self.scene_optimizer.correspondence_generator.generate_correspondences(
                client,
                self.loader.get_all_images_as_futures(client),
                image_pair_indices,
            )

            two_view_results_dict = run_two_view_estimator_as_futures(
                client,
                self.scene_optimizer.two_view_estimator,
                keypoints_list,
                putative_corr_idxs_dict,
                intrinsics,
                self.loader.get_relative_pose_priors(image_pair_indices),
                self.loader.get_gt_cameras(),
                gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
            )

        i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, two_view_reports_dict = unzip_two_view_results(two_view_results_dict)

        delayed_sfm_result, delayed_io = self.scene_optimizer.create_computation_graph(
            keypoints_list=keypoints_list,
            i2Ri1_dict=i2Ri1_dict,
            i2Ui1_dict=i2Ui1_dict,
            v_corr_idxs_dict=v_corr_idxs_dict,
            two_view_reports=two_view_reports_dict,
            num_images=len(self.loader),
            images=self.loader.create_computation_graph_for_images(),
            camera_intrinsics=intrinsics,
            relative_pose_priors=self.loader.get_relative_pose_priors(image_pair_indices),
            absolute_pose_priors=self.loader.get_absolute_pose_priors(),
            cameras_gt=self.loader.get_gt_cameras(),
            gt_wTi_list=self.loader.get_gt_poses(),
            gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
        )

        with performance_report(filename="scene-optimizer-dask-report.html"):
            sfm_result, *io = dask.compute(delayed_sfm_result, *delayed_io)

        assert isinstance(sfm_result, GtsfmData)

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)

        return sfm_result


def unzip_two_view_results(
    two_view_results: Dict[Tuple[int, int], TWO_VIEW_OUTPUT]
) -> Tuple[
    Dict[Tuple[int, int], Rot3],
    Dict[Tuple[int, int], Unit3],
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], TwoViewEstimationReport],
]:
    """Unzip the tuple TWO_VIEW_OUTPUT into 1 dictionary for 1 element in the tuple."""
    i2Ri1_dict: Dict[Tuple[int, int], Rot3] = {}
    i2Ui1_dict: Dict[Tuple[int, int], Unit3] = {}
    v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray] = {}
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport] = {}

    for (i1, i2), two_view_output in two_view_results.items():
        i2Ri1 = two_view_output[0]
        i2Ui1 = two_view_output[1]
        if i2Ri1 is None or i2Ui1 is None:
            continue

        i2Ri1_dict[(i1, i2)] = i2Ri1
        i2Ui1_dict[(i1, i2)] = i2Ui1
        v_corr_idxs_dict[(i1, i2)] = two_view_output[2]
        two_view_reports_dict[(i1, i2)] = two_view_output[5]

    return i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, two_view_reports_dict
