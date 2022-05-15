"""Runs GTSfM on an AstroNet dataset.

Author: Travis Driver
"""
import argparse
import time

import dask
from dask.distributed import Client, LocalCluster, performance_report

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.astronet_loader import AstronetLoader
from gtsfm.retriever.retriever_base import ImageMatchingRegime
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

    def run(self) -> None:
        """Run Structure-from-Motion (SfM) pipeline."""
        start_time = time.time()
        # Create dask client.
        cluster = LocalCluster(
            n_workers=self.parsed_args.num_workers, threads_per_worker=self.parsed_args.threads_per_worker
        )

        with Client(cluster) as client, performance_report(filename="dask-report.html"):
            pairs_graph = self.retriever.create_computation_graph(self.loader)
            image_pair_indices = pairs_graph.compute()

            # Scatter surface mesh across all nodes to preserve computation time and memory.
            gt_scene_trimesh_future = client.scatter(self.loader.gt_scene_trimesh, broadcast=True)

            # Prepare computation graph.
            delayed_sfm_result, delayed_io = self.scene_optimizer.create_computation_graph(
                num_images=len(self.loader),
                image_pair_indices=image_pair_indices,
                images_graph=dict(enumerate(self.loader.create_computation_graph_for_images())),
                all_intrinsics=self.loader.get_all_intrinsics(),
                image_shapes=self.loader.get_image_shapes(),
                gt_scene_mesh=gt_scene_trimesh_future,
                cameras_gt=self.loader.get_gt_cameras(),
                gt_wTi_list=self.loader.get_gt_poses(),
                matching_regime=ImageMatchingRegime(self.parsed_args.matching_regime),
                absolute_pose_priors=self.loader.get_absolute_pose_priors(),
                relative_pose_priors=self.loader.get_relative_pose_priors(),
            )

            # Run SfM pipeline.
            sfm_result, *io = dask.compute(delayed_sfm_result, *delayed_io)

        assert isinstance(sfm_result, GtsfmData)
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", (time.time() - start_time) / 60)


if __name__ == "__main__":
    runner = GtsfmRunnerAstronetLoader()
    runner.run()
