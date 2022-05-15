"""Run front-end on the hilti dataset without using Dask.

Authors: Ayush Baid
"""
import time
from argparse import ArgumentParser, Namespace

import hydra
from hydra.utils import instantiate

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.hilti_loader import HiltiLoader
from gtsfm.retriever.rig_retriever import RigRetriever
from gtsfm.scene_optimizer import SceneOptimizer

logger = logger_utils.get_logger()


class HiltiFrontendRunner:
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

    def run_frontend(self):
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

        counter = 0
        for i1, i2 in pairs_for_frontend:
            counter += 1
            if counter % 100 == 0:
                logger.info("%d/%d pairs", counter, len(pairs_for_frontend))
            if i1 in keypoints_dict and i2 in keypoints_dict:
                self.scene_optimizer.two_view_estimator.run_2view(
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

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info("GTSFM took %.2f minutes to compute frontend.", duration_sec / 60)


if __name__ == "__main__":
    runner = HiltiFrontendRunner()
    runner.run_frontend()
