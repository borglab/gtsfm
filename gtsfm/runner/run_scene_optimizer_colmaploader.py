import argparse

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

logger = logger_utils.get_logger()


class GtsfmRunnerColmapLoader(GtsfmRunnerBase):
    @property
    def tag(self) -> str:
        return "GTSFM with intrinsics and image names stored in COLMAP-format"

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super(GtsfmRunnerColmapLoader, self).construct_argparser()

        parser.add_argument(
            "--images_dir", type=str, required=True, help="path to directory containing png, jpeg, or jpg images files"
        )
        parser.add_argument(
            "--colmap_files_dirpath",
            type=str,
            required=True,
            help="path to directory containing images.txt, points3D.txt, and cameras.txt",
        )

        return parser

    def construct_loader(self) -> LoaderBase:
        loader = ColmapLoader(
            colmap_files_dirpath=self.parsed_args.colmap_files_dirpath,
            images_dir=self.parsed_args.images_dir,
            max_resolution=self.parsed_args.max_resolution,
        )

        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerColmapLoader()
    runner.run()
