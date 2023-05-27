"""Base class for runner that executes SfM."""

import argparse
import os
import time
from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import dask
import hydra
from dask.distributed import Client, LocalCluster, SSHCluster, performance_report
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
from gtsam import Rot3, Unit3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.frontend.correspondence_generator.det_desc_correspondence_generator import DetDescCorrespondenceGenerator
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import ImageMatchingRegime
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE, CAMERA_TYPE
from gtsfm.two_view_estimator import TwoViewEstimator, TWO_VIEW_OUTPUT, TwoViewEstimationReport
from gtsfm.common.pose_prior import PosePrior

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

        logger.info("\n\nSceneOptimizer: " + str(scene_optimizer))
        return scene_optimizer

    def apply_correspondence_generator(
        self,
        client: Client,
        images: List[Image],
        image_pairs: List[Tuple[int, int]],
        camera_intrinsics: List[CALIBRATION_TYPE],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        gt_cameras: List[Optional[CAMERA_TYPE]],
        gt_scene_mesh: Optional[Any],
        correspondence_generator: CorrespondenceGeneratorBase,
        two_view_estimator: TwoViewEstimator,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], TWO_VIEW_OUTPUT]]:
        assert isinstance(correspondence_generator, DetDescCorrespondenceGenerator)

        def apply_det_desc(det_desc: DetectorDescriptorBase, image: Image) -> Tuple[Keypoints, np.ndarray]:
            return det_desc.apply(image)

        def apply_matcher_and_two_view_estimator(
            feature_matcher: MatcherBase,
            two_view_estimator: TwoViewEstimator,
            features_i1: Tuple[Keypoints, np.ndarray],
            features_i2: Tuple[Keypoints, np.ndarray],
            im_shape_i1: Tuple[int, int],
            im_shape_i2: Tuple[int, int],
            camera_intrinsics_i1: CALIBRATION_TYPE,
            camera_intrinsics_i2: CALIBRATION_TYPE,
            i2Ti1_prior: Optional[PosePrior],
            gt_camera_i1: Optional[CAMERA_TYPE],
            gt_camera_i2: Optional[CAMERA_TYPE],
            gt_scene_mesh: Optional[Any] = None,
        ) -> TWO_VIEW_OUTPUT:
            putative_corr_idxs = feature_matcher.apply(
                features_i1[0], features_i2[0], features_i1[1], features_i2[1], im_shape_i1, im_shape_i2
            )

            return two_view_estimator.apply(
                features_i1[0],
                features_i2[0],
                putative_corr_idxs,
                camera_intrinsics_i1,
                camera_intrinsics_i2,
                i2Ti1_prior,
                gt_camera_i1,
                gt_camera_i2,
                gt_scene_mesh,
            )

        det_desc_future = client.scatter(correspondence_generator.detector_descriptor, broadcast=False)
        feature_matcher_future = client.scatter(correspondence_generator.matcher, broadcast=False)
        two_view_estimator_future = client.scatter(two_view_estimator, broadcast=False)
        features_futures = [client.submit(apply_det_desc, det_desc_future, image) for image in images]

        two_view_output_futures = {
            (i1, i2): client.submit(
                apply_matcher_and_two_view_estimator,
                feature_matcher_future,
                two_view_estimator_future,
                features_futures[i1],
                features_futures[i2],
                images[i1].shape,
                images[i2].shape,
                camera_intrinsics[i1],
                camera_intrinsics[i2],
                relative_pose_priors.get((i1, i2)),
                gt_cameras[i1],
                gt_cameras[i2],
                gt_scene_mesh,
            )
            for (i1, i2) in image_pairs
        }

        two_view_output_dict = client.gather(two_view_output_futures)
        keypoints_futures = [client.submit(lambda f: f[0], f) for f in features_futures]
        keypoints_list = client.gather(keypoints_futures)

        return keypoints_list, two_view_output_dict

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
        if isinstance(self.scene_optimizer.correspondence_generator, ImageCorrespondenceGenerator):
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

        # TODO: Use futures
        image_pair_indices = self.scene_optimizer.retriever.apply(
            self.loader, plots_output_dir=self.scene_optimizer._plot_base_path
        )

        images = [self.loader.get_image(i) for i in range(len(self.loader))]
        intrinsics = [self.loader.get_camera_intrinsics(i) for i in range(len(self.loader))]

        with Client(cluster) as client, performance_report(filename="submit-perf.html"):
            keypoints_list, two_view_results_dict = self.apply_correspondence_generator(
                client,
                images,
                image_pair_indices,
                intrinsics,
                self.loader.get_relative_pose_priors(image_pair_indices),
                self.loader.get_gt_cameras(),
                gt_scene_mesh=None,
                correspondence_generator=self.scene_optimizer.correspondence_generator,
                two_view_estimator=self.scene_optimizer.two_view_estimator,
            )

        i2Ri1_dict: Dict[Tuple[int, int], Rot3] = {}
        i2Ui1_dict: Dict[Tuple[int, int], Unit3] = {}
        v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray] = {}
        two_view_reports_post_isp: Dict[Tuple[int, int], TwoViewEstimationReport] = {}

        for (i1, i2), two_view_output in two_view_results_dict.items():
            i2Ri1 = two_view_output[0]
            i2Ui1 = two_view_output[1]
            if i2Ri1 is None:
                continue

            assert i2Ui1
            i2Ri1_dict[(i1, i2)] = i2Ri1
            i2Ui1_dict[(i1, i2)] = i2Ui1
            v_corr_idxs_dict[(i1, i2)] = two_view_output[2]
            two_view_reports_post_isp[(i1, i2)] = two_view_output[5]

        delayed_sfm_result, delayed_io = self.scene_optimizer.create_computation_graph(
            keypoints_list=keypoints_list,
            i2Ri1_dict=i2Ri1_dict,
            i2Ui1_dict=i2Ui1_dict,
            v_corr_idxs_dict=v_corr_idxs_dict,
            two_view_reports=two_view_reports_post_isp,
            num_images=len(self.loader),
            image_pair_indices=image_pair_indices,
            image_graph=self.loader.create_computation_graph_for_images(),
            camera_intrinsics=self.loader.create_computation_graph_for_intrinsics(),
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
