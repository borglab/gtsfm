import argparse
import os
from pathlib import Path

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"

logger = logger_utils.get_logger()


class GtsfmRunnerOlssonLoader(GtsfmRunnerBase):
    @property
    def tag(self) -> str:
        return "GTSFM on Dataset in Olsson's Lund format"

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super(GtsfmRunnerOlssonLoader, self).construct_argparser()

        parser.add_argument("--dataset_root", type=str, default=os.path.join(DATA_ROOT, "set1_lund_door"), help="")

        return parser

    def construct_loader(self) -> LoaderBase:
        loader = OlssonLoader(
            self.parsed_args.dataset_root,
            max_frame_lookahead=self.parsed_args.max_frame_lookahead,
            max_resolution=self.parsed_args.max_resolution,
        )

        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerOlssonLoader()
    runner.run()
