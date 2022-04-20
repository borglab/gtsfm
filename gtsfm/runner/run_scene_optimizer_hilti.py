import argparse

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.hilti_loader import HiltiLoader
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

logger = logger_utils.get_logger()


class GtsfmRunnerHiltiLoader(GtsfmRunnerBase):
    def __init__(self):
        super(GtsfmRunnerHiltiLoader, self).__init__(tag="GTSFM for the hilti loader")

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super(GtsfmRunnerHiltiLoader, self).construct_argparser()

        parser.add_argument(
            "--dataset_dirpath",
            type=str,
            required=True,
            help="path to directory containing the calibration files and the images",
        )

        parser.add_argument("--step_size", type=int, default=10, help="Step size between timestamps")

        parser.add_argument("--max_length", type=int, default=50, help="Max number of timestamps to process")

        return parser

    def construct_loader(self) -> LoaderBase:
        loader = HiltiLoader(
            base_folder=self.parsed_args.dataset_dirpath,
            cams_to_use={0, 1, 2, 3, 4},
            max_frame_lookahead=self.parsed_args.max_frame_lookahead,
            max_resolution=self.parsed_args.max_resolution,
            step_size=self.parsed_args.step_size,
            max_length=self.parsed_args.max_length,
        )

        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerHiltiLoader()
    runner.run()
