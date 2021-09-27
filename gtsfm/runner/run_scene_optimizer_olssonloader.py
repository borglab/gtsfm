"""Runs Gtsfm using OlssonLoader.

Author: Ayush Baid
"""
import os
from argparse import ArgumentParser
from pathlib import Path


import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"

logger = logger_utils.get_logger()


class GtsfmRunnerOlssonLoader(GtsfmRunnerBase):
    """Gtsfm runner for OlssonLoader."""

    def __init__(self):
        """Initializes the class by calling the base class' constructor with a tag."""
        super(GtsfmRunnerOlssonLoader, self).__init__(tag="GTSFM on Dataset in Olsson's Lund format")

    def construct_argparser(self) -> ArgumentParser:
        """Constructs the argparser by using the super class implementation and adding OlssonLoader specific args."""
        parser = super(GtsfmRunnerOlssonLoader, self).construct_argparser()

        parser.add_argument("--dataset_root", type=str, default=os.path.join(DATA_ROOT, "set1_lund_door"), help="")
        parser.add_argument("--image_extension", type=str, default="JPG", help="")

        return parser

    def construct_loader(self) -> LoaderBase:
        """Constructs the OlssonLoader."""
        loader = OlssonLoader(
            self.parsed_args.dataset_root,
            image_extension=self.parsed_args.image_extension,
            max_frame_lookahead=self.parsed_args.max_frame_lookahead,
            max_resolution=self.parsed_args.max_resolution,
        )

        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerOlssonLoader()
    runner.run()
