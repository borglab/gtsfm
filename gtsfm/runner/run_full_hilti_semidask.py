"""Run front-end on the hilti dataset without using Dask.

Authors: Ayush Baid
"""
import time
from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, Tuple

import dask
import hydra
import numpy as np
from dask.distributed import LocalCluster, Client
from hydra.utils import instantiate
from gtsam import Rot3, Unit3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.loader.hilti_loader import HiltiLoader
from gtsfm.retriever.rig_retriever import RigRetriever
from gtsfm.scene_optimizer import SceneOptimizer

logger = logger_utils.get_logger()


class HiltiRunner:
    def __init__(self) -> None:
        argparser: ArgumentParser = self.construct_argparser()
        parsed_args: Namespace = argparser.parse_args()

        self.loader = HiltiLoader(
            base_folder=parsed_args.dataset_dirpath, max_length=parsed_args.max_length, subsample=parsed_args.subsample
        )

        self.retriever = RigRetriever(subsample=parsed_args.subsample, threshold=parsed_args.proxy_threshold)

        self.dask_cluster = LocalCluster(
            n_workers=parsed_args.num_workers,
            threads_per_worker=parsed_args.threads_per_worker,
        )

        with hydra.initialize_config_module(config_module="gtsfm.configs"):
            # config is relative to the gtsfm module
            cfg = hydra.compose(
                config_name=parsed_args.config_name,
            )
            self.scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

    def construct_argparser(self) -> ArgumentParser:
        parser = ArgumentParser(description="Run frontend for the hilti dataset")

        parser.add_argument(
            "--dataset_dirpath",
            type=str,
            required=True,
            help="path to directory containing the calibration files and the images",
        )
        parser.add_argument(
            "--config_name",
            type=str,
            default="deep_front_end_hilti.yaml",
            help="Choose the config file",
        )
        parser.add_argument(
            "--proxy_threshold",
            type=int,
            default=100,
            help="amount of 'proxy' correspondences that will trigger an image-pair. Default 100.",
        )
        parser.add_argument("--max_length", type=int, default=None, help="Max number of timestamps to process")
        parser.add_argument(
            "--subsample",
            type=int,
            default=6,
            help="Subsample the timestamps by given value n (pick every nth rig for visual SfM)",
        )
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

        return parser

    def run_frontend(
        self,
    ) -> Tuple[
        Dict[int, Keypoints],
        Dict[Tuple[int, int], Optional[Rot3]],
        Dict[Tuple[int, int], Optional[Unit3]],
        Dict[Tuple[int, int], np.ndarray],
    ]:
        start_time = time.time()

        pairs_for_frontend = self.retriever.run(self.loader)

        image_indices_for_feature_extraction = list(sum(pairs_for_frontend, ()))
        keypoints_dict = {}
        descriptors_dict = {}
        image_shapes = {}

        for i in image_indices_for_feature_extraction:
            image = self.loader.get_image(i)
            if image is not None:
                keypoints, descriptors = self.scene_optimizer.feature_extractor.detector_descriptor.detect_and_describe(
                    image
                )
                keypoints_dict[i] = keypoints
                descriptors_dict[i] = descriptors
                image_shapes[i] = (image.height, image.width)

        i2Ri1_dict = {}
        i2Ui1_dict = {}
        v_corr_idxs_dict = {}
        counter = 0
        for i1, i2 in pairs_for_frontend:
            counter += 1
            if counter % 100 == 0:
                logger.info("%d/%d pairs", counter, len(pairs_for_frontend))
            if i1 in keypoints_dict and i2 in keypoints_dict:
                i2Ri1, i2Ui1, v_corr_idxs = self.scene_optimizer.two_view_estimator.run_2view(
                    keypoints_i1=keypoints_dict[i1],
                    keypoints_i2=keypoints_dict[i2],
                    descriptors_i1=descriptors_dict[i1],
                    descriptors_i2=descriptors_dict[i2],
                    camera_intrinsics_i1=self.loader.get_camera_intrinsics(i1),
                    camera_intrinsics_i2=self.loader.get_camera_intrinsics(i2),
                    im_shape_i1=image_shapes[i1],
                    im_shape_i2=image_shapes[i2],
                    i2Ti1_prior=None,
                    gt_wTi1=None,
                    gt_wTi2=None,
                    gt_scene_mesh=None,
                )

                i2Ri1_dict[(i1, i2)] = i2Ri1
                i2Ui1_dict[(i1, i2)] = i2Ui1
                v_corr_idxs_dict[(i1, i2)] = v_corr_idxs

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute frontend.", duration_sec / 60)

        return keypoints_dict, i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict

    def run_full(self):
        start_time = time.time()
        keypoints_dict, i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict = self.run_frontend()

        num_images = len(self.loader)
        keypoints_list = [keypoints_dict.get(i, Keypoints(np.array([[]]))) for i in range(num_images)]

        delayed_sfm_result, delayed_io = self.scene_optimizer.create_computation_graph_for_backend(
            num_images=len(self.loader),
            delayed_keypoints=keypoints_list,
            i2Ri1_dict=i2Ri1_dict,
            i2Ui1_dict=i2Ui1_dict,
            v_corr_idxs_dict=v_corr_idxs_dict,
            image_graph=self.loader.create_computation_graph_for_images(),
            all_intrinsics=self.loader.get_all_intrinsics(),
            relative_pose_priors=self.loader.get_relative_pose_priors(),
            absolute_pose_priors=self.loader.get_absolute_pose_priors(),
            cameras_gt=self.loader.get_gt_cameras(),
            gt_wTi_list=self.loader.get_gt_poses(),
        )

        with Client(self.dask_cluster):
            sfm_result, *io = dask.compute(delayed_sfm_result, *delayed_io)

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)


if __name__ == "__main__":
    runner = HiltiRunner()
    runner.run_full()
