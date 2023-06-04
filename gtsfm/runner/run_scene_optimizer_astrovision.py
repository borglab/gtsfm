"""Runs GTSfM on an AstroVision dataset.

Author: Travis Driver
"""
import argparse

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.astrovision_loader import AstrovisionLoader
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

logger = logger_utils.get_logger()


class GtsfmRunnerAstrovisionLoader(GtsfmRunnerBase):
    def __init__(self):
        super(GtsfmRunnerAstrovisionLoader, self).__init__(tag="Run GTSfM on AstroVision segment")

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super(GtsfmRunnerAstrovisionLoader, self).construct_argparser()

        parser.add_argument(
            "--data_dir", "-d", type=str, required=True, help="path to directory containing AstroVision segment"
        )
        parser.add_argument(
            "--scene_mesh_path",
            "-m",
            type=str,
            default=None,
            help="Path to file containing triangular surface mesh of target body.",
        )

        return parser

    def construct_loader(self) -> LoaderBase:
        """Initialize loader."""
        loader = AstrovisionLoader(
            data_dir=self.parsed_args.data_dir,
            use_gt_extrinsics=True,
            use_gt_sfmtracks=False,
            max_frame_lookahead=self.parsed_args.max_frame_lookahead,
            max_resolution=self.parsed_args.max_resolution,
            gt_scene_mesh_path=self.parsed_args.scene_mesh_path,
        )

        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerAstrovisionLoader()
    runner.run()
