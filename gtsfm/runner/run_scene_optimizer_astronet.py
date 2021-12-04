"""Runs GTSfM on an AstroNet dataset.

Author: Travis Driver
"""
import argparse
import time

from dask.distributed import Client, LocalCluster, performance_report

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.astronet_loader import AstronetLoader
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

logger = logger_utils.get_logger()


class GtsfmRunnerAstronetLoader(GtsfmRunnerBase):
    def __init__(self):
        super(GtsfmRunnerAstronetLoader, self).__init__(tag="Run GTSfM on AstroNet segment")

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super(GtsfmRunnerAstronetLoader, self).construct_argparser()

        parser.add_argument(
            "--data_dir", "-d", type=str, required=True, help="path to directory containing AstroNet segment"
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
        loader = AstronetLoader(
            data_dir=self.parsed_args.data_dir,
            use_gt_extrinsics=True,
            use_gt_sfmtracks=False,
            max_frame_lookahead=self.parsed_args.max_frame_lookahead,
            max_resolution=self.parsed_args.max_resolution,
            gt_scene_mesh_path=self.parsed_args.scene_mesh_path,
        )

        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerAstronetLoader()
    runner.run()
