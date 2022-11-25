"""Runner for datasets used in 1DSFM and Colmap papers.

Authors: Yanwei Du
"""
import argparse

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.one_d_sfm_loader import OneDSFMLoader
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

logger = logger_utils.get_logger()


class GtsfmRunnerOneDSFMLoader(GtsfmRunnerBase):
    """Runner for datasets used in 1DSFM and Colmap papers."""

    def __init__(self):
        super(GtsfmRunnerOneDSFMLoader, self).__init__(tag="Run GTSFM on dataset from 1DSFM.")

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super(GtsfmRunnerOneDSFMLoader, self).construct_argparser()
        parser.add_argument("--dataset_root", type=str, default="", help="")
        parser.add_argument("--image_extension", type=str, default="jpg", help="")
        parser.add_argument(
            "--max_num_imgs", type=int, default=0, help="max number of image to process, default: 0(all)"
        )
        return parser

    def construct_loader(self) -> LoaderBase:
        loader = OneDSFMLoader(
            folder=self.parsed_args.dataset_root,
            image_extension=self.parsed_args.image_extension,
            max_num_imgs=self.parsed_args.max_num_imgs,
        )
        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerOneDSFMLoader()
    runner.run()
