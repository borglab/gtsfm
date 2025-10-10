"""Runner for the hilti dataset.

Authors: Ayush Baid
"""

import argparse

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.hilti_loader import HiltiLoader
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

logger = logger_utils.get_logger()


class GtsfmRunnerHiltiLoader(GtsfmRunnerBase):
    @property
    def tag(self) -> str:
        return "GTSFM for the hilti loader"

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super(GtsfmRunnerHiltiLoader, self).construct_argparser()

        parser.add_argument(
            "--dataset_dirpath",
            type=str,
            required=True,
            help="path to directory containing the calibration files and the images",
        )
        parser.add_argument(
            "--proxy_threshold",
            type=int,
            default=100,
            help="amount of 'proxy' correspondences that will trigger an image-pair. Default 100.",
        )

        parser.add_argument("--max_length", type=int, default=None, help="Max number of timestamps to process")

        return parser

    def construct_loader(self) -> LoaderBase:
        loader = HiltiLoader(
            base_folder=self.parsed_args.dataset_dirpath,
            max_length=self.parsed_args.max_length,
        )

        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerHiltiLoader()
    runner.run()
