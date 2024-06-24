"""Runs GTSfM on a Tanks and Temples dataset, using synthetic correspondences.

Author: John Lambert
"""

import argparse

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.tanks_and_temples_loader import TanksAndTemplesLoader
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

logger = logger_utils.get_logger()


# TODO(johnwlambert,travisdriver): Make this generic for any dataset with a GT mesh.
class GtsfmRunnerSyntheticTanksAndTemplesLoader(GtsfmRunnerBase):
    tag = "GTSFM with LiDAR scans, COLMAP camera poses, and image names stored in Tanks and Temples format"

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super().construct_argparser()

        parser.add_argument(
            "--dataset_root", type=str, required=True, help="Path to dir, for unzipped file containing packaged data."
        )
        parser.add_argument("--scene_name", type=str, required=True, help="Name of dataset scene.")
        parser.add_argument(
            "--max_num_images",
            type=int,
            default=None,
            help="Optionally specifies the maximum number of images from the dataset to use for reconstruction.",
        )
        return parser

    def construct_loader(self) -> LoaderBase:
        dataset_root = self.parsed_args.dataset_root
        scene_name = self.parsed_args.scene_name

        img_dir = f"{dataset_root}/{scene_name}"
        poses_fpath = f"{dataset_root}/{scene_name}_COLMAP_SfM.log"
        lidar_ply_fpath = f"{dataset_root}/{scene_name}.ply"
        colmap_ply_fpath = f"{dataset_root}/{scene_name}_COLMAP.ply"
        ply_alignment_fpath = f"{dataset_root}/{scene_name}_trans.txt"
        bounding_polyhedron_json_fpath = f"{dataset_root}/{scene_name}.json"
        loader = TanksAndTemplesLoader(
            img_dir=img_dir,
            poses_fpath=poses_fpath,
            lidar_ply_fpath=lidar_ply_fpath,
            ply_alignment_fpath=ply_alignment_fpath,
            bounding_polyhedron_json_fpath=bounding_polyhedron_json_fpath,
            colmap_ply_fpath=colmap_ply_fpath,
            # NOTE: Native resolution for T&T is 1080 px.
            max_resolution=self.parsed_args.max_resolution,
            max_num_images=self.parsed_args.max_num_images,
        )
        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerSyntheticTanksAndTemplesLoader()
    runner.run()
