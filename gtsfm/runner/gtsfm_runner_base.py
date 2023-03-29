"""Base class for runner that executes SfM."""

import argparse
import time
from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import dask
import hydra
from dask.distributed import Client, LocalCluster
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
from gtsam import Pose3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.frontend.correspondence_generator.det_desc_correspondence_generator import DetDescCorrespondenceGenerator
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.exhaustive_retriever import ExhaustiveRetriever
from gtsfm.retriever.joint_netvlad_sequential_retriever import JointNetVLADSequentialRetriever
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever
from gtsfm.retriever.retriever_base import ImageMatchingRegime, RetrieverBase
from gtsfm.retriever.rig_retriever import RigRetriever
from gtsfm.retriever.sequential_hilti_retriever import SequentialHiltiRetriever
from gtsfm.retriever.sequential_retriever import SequentialRetriever
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.two_view_estimator import TwoViewEstimator, TWO_VIEW_OUTPUT
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
                "retrieval",
                "sequential",
                "sequential_with_retrieval",
                "sequential_hilti",
                "rig_hilti",
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

        if self.parsed_args.correspondence_generator_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.correspondence"):
                correspondence_cfg = hydra.compose(
                    config_name=self.parsed_args.correspondence_generator_config_name,
                )
                logger.info("\n\nCorrespondenceGenerator override: " + OmegaConf.to_yaml(correspondence_cfg))
                scene_optimizer.correspondence_generator: CorrespondenceGeneratorBase = instantiate(
                    correspondence_cfg.CorrespondenceGenerator
                )

        if self.parsed_args.verifier_config_name is not None:
            with hydra.initialize_config_module(config_module="gtsfm.configs.verifier"):
                verifier_cfg = hydra.compose(
                    config_name=self.parsed_args.verifier_config_name,
                )
                logger.info("\n\nVerifier override: " + OmegaConf.to_yaml(verifier_cfg))
                scene_optimizer.two_view_estimator._verifier: VerifierBase = instantiate(verifier_cfg.verifier)

        if self.parsed_args.mvs_off:
            scene_optimizer.run_dense_optimizer = False

        logger.info("\n\nSceneOptimizer: " + str(scene_optimizer))
        return scene_optimizer

    def construct_retriever(self) -> RetrieverBase:
        """Set up retriever module."""
        matching_regime = ImageMatchingRegime(self.parsed_args.matching_regime)

        if matching_regime == ImageMatchingRegime.EXHAUSTIVE:
            retriever = ExhaustiveRetriever()

        elif matching_regime == ImageMatchingRegime.RETRIEVAL:
            retriever = NetVLADRetriever(num_matched=self.parsed_args.num_matched)

        elif matching_regime == ImageMatchingRegime.SEQUENTIAL:
            retriever = SequentialRetriever(max_frame_lookahead=self.parsed_args.max_frame_lookahead)

        elif matching_regime == ImageMatchingRegime.SEQUENTIAL_HILTI:
            retriever = SequentialHiltiRetriever(max_frame_lookahead=self.parsed_args.max_frame_lookahead)

        elif matching_regime == ImageMatchingRegime.RIG_HILTI:
            retriever = RigRetriever(threshold=self.parsed_args.proxy_threshold)

        elif matching_regime == ImageMatchingRegime.SEQUENTIAL_WITH_RETRIEVAL:
            retriever = JointNetVLADSequentialRetriever(
                num_matched=self.parsed_args.num_matched, max_frame_lookahead=self.parsed_args.max_frame_lookahead
            )
        return retriever

    def apply_correspondence_generator(
        self,
        client: Client,
        images: List[Image],
        image_pairs: List[Tuple[int, int]],
        camera_intrinsics: List[CALIBRATION_TYPE],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        gt_poses: List[Optional[Pose3]],
        gt_scene_mesh: Optional[Any],
        correspondence_generator: CorrespondenceGeneratorBase,
        two_view_estimator: VerifierBase,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], TWO_VIEW_OUTPUT]]:
        assert isinstance(correspondence_generator, DetDescCorrespondenceGenerator)

        def apply_det_desc(det_desc: DetectorDescriptorBase, image: Image) -> Tuple[Keypoints, np.ndarray]:
            return det_desc.apply(image)

        def apply_matcher(
            feature_matcher: MatcherBase,
            features_i1: Tuple[Keypoints, np.ndarray],
            features_i2: Tuple[Keypoints, np.ndarray],
            im_shape_i1: Tuple[int, int],
            im_shape_i2: Tuple[int, int],
        ) -> np.ndarray:
            return feature_matcher.apply(
                features_i1[0], features_i2[0], features_i1[1], features_i2[1], im_shape_i1, im_shape_i2
            )

        def apply_two_view_estimator(
            two_view_estimator: TwoViewEstimator,
            features_i1: Tuple[Keypoints, np.ndarray],
            features_i2: Tuple[Keypoints, np.ndarray],
            match_indices: np.ndarray,
            camera_intrinsics_i1: CALIBRATION_TYPE,
            camera_intrinsics_i2: CALIBRATION_TYPE,
            im_shape_i1: Tuple[int, int],
            im_shape_i2: Tuple[int, int],
            i2Ti1_prior: Optional[PosePrior],
            gt_wTi1: Optional[Pose3],
            gt_wTi2: Optional[Pose3],
            gt_scene_mesh: Optional[Any] = None,
        ) -> Dict[Tuple[int, int], TWO_VIEW_OUTPUT]:
            return two_view_estimator.apply(
                features_i1[0],
                features_i2[0],
                match_indices,
                camera_intrinsics_i1,
                camera_intrinsics_i2,
                im_shape_i1,
                im_shape_i2,
                i2Ti1_prior,
                gt_wTi1,
                gt_wTi2,
                gt_scene_mesh,
            )

        det_desc_future = client.scatter(correspondence_generator.det_desc, broadcast=True)
        feature_matcher_future = client.scatter(correspondence_generator.matcher, broadcast=True)
        features_futures = [client.submit(apply_det_desc, det_desc_future, image) for image in images]
        matches_futures = {
            (i1, i2): client.submit(
                apply_matcher,
                feature_matcher_future,
                features_futures[i1],
                features_futures[i2],
                images[i1].shape,
                images[i2].shape,
            )
            for (i1, i2) in image_pairs
        }

        two_view_output_futures = {
            (i1, i2): client.submit(
                apply_two_view_estimator,
                two_view_estimator,
                features_futures[i1],
                features_futures[i2],
                matches_futures[(i1, i2)],
                camera_intrinsics[i1],
                camera_intrinsics[i2],
                images[i1].shape,
                images[i2].shape,
                relative_pose_priors.get((i1, i2)),
                gt_poses[i1],
                gt_poses[i2],
                gt_scene_mesh,
            )
            for (i1, i2) in image_pairs
        }

        features_list, two_view_output_dict = client.gather((features_futures, two_view_output_futures))
        keypoints_list = [f[0] for f in features_list]

        return keypoints_list, two_view_output_dict

    def run(self) -> None:
        """Run the SceneOptimizer."""
        start_time = time.time()

        # create dask client
        cluster = LocalCluster(
            n_workers=self.parsed_args.num_workers, threads_per_worker=self.parsed_args.threads_per_worker
        )
        client = Client(cluster)

        # create process graph
        process_graph_generator = ProcessGraphGenerator()
        if isinstance(self.scene_optimizer.correspondence_generator, ImageCorrespondenceGenerator):
            process_graph_generator.is_image_correspondence = True
        process_graph_generator.save_graph()

        images = [self.loader.get_image_full_res(i) for i in range(len(self.loader))]
        image_pair_indices = self.retriever.apply(self.loader)

        keypoints_list, two_view_results_dict = self.apply_correspondence_generator(
            client,
            images,
            image_pair_indices,
            self.loader.get_all_intrinsics(),
            self.loader.get_relative_pose_priors(image_pair_indices),
            self.loader.get_gt_poses(),
            gt_scene_mesh=None,
            correspondence_generator=self.scene_optimizer.correspondence_generator,
            two_view_estimator=self.scene_optimizer.two_view_estimator,
        )

        sfm_result = self.scene_optimizer.apply_multiview_estimator(
            keypoints_list=keypoints_list,
            two_view_estimator_results=two_view_results_dict,
            num_images=len(self.loader),
            image_pair_indices=image_pair_indices,
            images=images,
            all_intrinsics=self.loader.get_all_intrinsics(),
            relative_pose_priors=self.loader.get_relative_pose_priors(image_pair_indices),
            absolute_pose_priors=self.loader.get_absolute_pose_priors(),
            cameras_gt=self.loader.get_gt_cameras(),
            gt_wTi_list=self.loader.get_gt_poses(),
            matching_regime=ImageMatchingRegime(self.parsed_args.matching_regime),
        )

        assert isinstance(sfm_result, GtsfmData)

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)
