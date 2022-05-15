"""Run front-end on the hilti dataset without using Dask.

Authors: Ayush Baid
"""
import time
from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
from hydra.utils import instantiate
from gtsam import Rot3, Unit3

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.loader.hilti_loader import HiltiLoader
from gtsfm.retriever.rig_retriever import RigRetriever
from gtsfm.scene_optimizer import SceneOptimizer, save_gtsfm_data, save_metrics_reports, save_visualizations
from gtsfm.multi_view_optimizer import init_cameras


logger = logger_utils.get_logger()


class HiltiRunner:
    def __init__(self) -> None:
        argparser: ArgumentParser = self.construct_argparser()
        parsed_args: Namespace = argparser.parse_args()

        self.loader = HiltiLoader(
            base_folder=parsed_args.dataset_dirpath, max_length=parsed_args.max_length, subsample=parsed_args.subsample
        )

        self.retriever = RigRetriever(subsample=parsed_args.subsample, threshold=parsed_args.proxy_threshold)

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

        return parser

    def run_frontend(
        self,
    ) -> Tuple[
        Dict[int, Keypoints],
        Dict[Tuple[int, int], Optional[Rot3]],
        Dict[Tuple[int, int], Optional[Unit3]],
        Dict[Tuple[int, int], np.ndarray],
        Dict[int, Tuple[int, int]],
    ]:
        start_time = time.time()

        pairs_for_frontend = self.retriever.run(self.loader)

        image_indices_for_feature_extraction = set(sum(pairs_for_frontend, ()))
        keypoints_dict = {}
        descriptors_dict = {}
        image_shapes = {}

        counter = 0
        for i in image_indices_for_feature_extraction:
            counter += 1
            if counter % 20 == 0:
                logger.info("%d/%d images", counter, len(image_indices_for_feature_extraction))

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

        return keypoints_dict, i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, image_shapes

    def run_full(self):
        start_time = time.time()
        keypoints_dict, i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, image_shapes = self.run_frontend()

        num_images = len(self.loader)
        keypoints_list = [keypoints_dict.get(i, Keypoints(np.array([[]]))) for i in range(num_images)]
        image_shapes_list = [image_shapes.get(i, (100, 100)) for i in range(num_images)]

        relative_pose_priors = self.loader.get_relative_pose_priors()
        absolute_pose_priors = self.loader.get_absolute_pose_priors()
        all_intrinsics = self.loader.get_all_intrinsics()
        gt_wTi_list = self.loader.get_gt_poses()
        gt_cameras = self.loader.get_gt_cameras()

        pruned_i2Ri1_dict, pruned_i2Ui1_dict = graph_utils.prune_to_largest_connected_component(
            i2Ri1_dict, i2Ui1_dict, relative_pose_priors
        )

        wRi = self.scene_optimizer.multiview_optimizer.rot_avg_module.run_rotation_averaging(
            num_images, pruned_i2Ri1_dict, relative_pose_priors
        )
        rot_avg_metrics = self.scene_optimizer.multiview_optimizer.rot_avg_module.evaluate(
            wRi_computed=wRi, wTi_gt=gt_wTi_list
        )

        wti, ta_metrics = self.scene_optimizer.multiview_optimizer.trans_avg_module.run_translation_averaging(
            num_images=num_images,
            i2Ui1_dict=pruned_i2Ui1_dict,
            wRi_list=wRi,
            absolute_pose_priors=absolute_pose_priors,
            scale_factor=1,
            gt_wTi_list=gt_wTi_list,
        )

        initialized_cameras = init_cameras(wRi_list=wRi, wti_list=wti, intrinsics_list=all_intrinsics)

        ba_input, da_metrics = self.scene_optimizer.multiview_optimizer.data_association_module.run_da(
            num_images=num_images,
            cameras=initialized_cameras,
            corr_idxs_dict=v_corr_idxs_dict,
            keypoints_list=keypoints_list,
            cameras_gt=gt_cameras,
            relative_pose_priors=relative_pose_priors,
            images=None,
        )

        ba_unfiltered, ba_output, _ = self.scene_optimizer.multiview_optimizer.ba_optimizer.run_ba(
            initial_data=ba_input,
            absolute_pose_priors=absolute_pose_priors,
            relative_pose_priors=relative_pose_priors,
            verbose=True,
            intrinsincs=all_intrinsics,
        )
        ba_metrics = self.scene_optimizer.multiview_optimizer.ba_optimizer.evaluate(
            ba_unfiltered, ba_output, gt_cameras
        )

        ba_input_aligned = ba_input.align_via_Sim3_to_poses(wTi_list_ref=gt_wTi_list)
        ba_output_aligned = ba_output.align_via_Sim3_to_poses(wTi_list_ref=gt_wTi_list)

        save_visualizations(ba_input_aligned, ba_output_aligned, gt_wTi_list)
        save_gtsfm_data(None, image_shapes_list, self.loader.get_image_fnames(), ba_input, ba_output)

        save_metrics_reports([rot_avg_metrics, ta_metrics, da_metrics, ba_metrics])

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", duration_sec / 60)


if __name__ == "__main__":
    runner = HiltiRunner()
    runner.run_full()
