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

    @property
    def tag(self) -> str:
        return "Run GTSFM on dataset from 1DSFM."

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super(GtsfmRunnerOneDSFMLoader, self).construct_argparser()
        parser.add_argument("--dataset_dir", type=str, default="", help="")
        parser.add_argument("--enable_no_exif", action="store_true", help="")
        parser.add_argument("--default_focal_length_factor", type=float, default=1.2, help="")
        return parser

    def construct_loader(self) -> LoaderBase:
        loader = OneDSFMLoader(
            folder=self.parsed_args.dataset_dir,
            enable_no_exif=self.parsed_args.enable_no_exif,
            default_focal_length_factor=self.parsed_args.default_focal_length_factor,
        )
        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerOneDSFMLoader()
    runner.run()
