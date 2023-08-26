"""Runner for datasets loaded from the MobileBrick loader.

Authors: Akshay Krishnan
"""
import argparse

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.mobilebrick_loader import MobilebrickLoader
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

logger = logger_utils.get_logger()


class GtsfmRunnerMobilebrickLoader(GtsfmRunnerBase):
    """Runner for the Mobilebrick dataset."""

    tag = "Run GTSFM on dataset from MobileBrick."

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super(GtsfmRunnerMobilebrickLoader, self).construct_argparser()
        parser.add_argument("--data_dir", type=str, default="", help="")
        parser.add_argument("--use_gt_intrinsics", type=bool, default=False, help="")
        return parser

    def construct_loader(self) -> LoaderBase:
        loader = MobilebrickLoader(
            data_dir=self.parsed_args.data_dir,
            use_gt_intrinsics=self.parsed_args.use_gt_intrinsics,
            max_frame_lookahead=self.parsed_args.max_frame_lookahead,
            max_resolution=self.parsed_args.max_resolution,
        )
        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerMobilebrickLoader()
    runner.run()
