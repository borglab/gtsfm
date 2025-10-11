from argparse import ArgumentParser

from gtsfm.loader.argoverse_dataset_loader import ArgoverseDatasetLoader
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase


class GtsfmRunnerArgoverse(GtsfmRunnerBase):
    """Runner for datasets from Argoverse self-driving dataset"""

    @property
    def tag(self) -> str:
        return "Run GTSFM on scene from Argoverse."

    def construct_argparser(self) -> ArgumentParser:
        parser = super().construct_argparser()

        parser.add_argument(
            "--log_id",
            default="273c1883-673a-36bf-b124-88311b1a80be",
            type=str,
            help="unique ID of Argoverse vehicle log",
        )
        parser.add_argument(
            "--dataset_dir",
            default="/srv/share/cliu324/argoverse-tracking-readonly/train1",
            type=str,
            help="directory where raw Argoverse logs are stored on disk",
        )
        parser.add_argument(
            "--camera_name",
            default="ring_front_center",
            type=str,
            help="Which of 9 Argoverse cameras",
        )
        parser.add_argument(
            "--stride",
            default=10,
            type=int,
            help="image sub-sampling interval, e.g. every 2 images, every 4 images, etc.",
        )
        parser.add_argument(
            "--max_num_imgs",
            default=20,
            type=int,
            help="maximum number of images to include in dataset (starting from beginning of log sequence)",
        )
        parser.add_argument(
            "--max_lookahead_sec",
            default=2,
            type=float,
            help="",
        )

        return parser

    def construct_loader(self) -> LoaderBase:
        return ArgoverseDatasetLoader(
            dataset_dir=self.parsed_args.dataset_dir,
            log_id=self.parsed_args.log_id,
            stride=self.parsed_args.stride,
            max_num_imgs=self.parsed_args.max_num_imgs,
            max_lookahead_sec=self.parsed_args.max_lookahead_sec,
            camera_name=self.parsed_args.camera_name,
            max_resolution=self.parsed_args.max_resolution,
        )
