"""Runs GTSfM on an AstroVision dataset.

Author: Travis Driver
"""
import argparse
import time

import dask
from dask.distributed import Client, LocalCluster, performance_report

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.astrovision_loader import AstrovisionLoader
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

    def run(self) -> None:
        """Run Structure-from-Motion (SfM) pipeline."""
        start_time = time.time()
        # Create dask client.
        cluster = LocalCluster(
            n_workers=self.parsed_args.num_workers, threads_per_worker=self.parsed_args.threads_per_worker
        )

        with Client(cluster) as client, performance_report(filename="dask-report.html"):
            pairs_graph = self.scene_optimizer.retriever.create_computation_graph(
                self.loader, self.scene_optimizer._plot_base_path
            )
            image_pair_indices = pairs_graph.compute()

            (
                delayed_keypoints,
                delayed_putative_corr_idxs_dict,
            ) = self.scene_optimizer.correspondence_generator.create_computation_graph(
                delayed_images=self.loader.create_computation_graph_for_images(),
                delayed_image_shapes=self.loader.create_computation_graph_for_image_shapes(),
                image_pair_indices=image_pair_indices,
            )
            keypoints_list, putative_corr_idxs_dict = dask.compute(delayed_keypoints, delayed_putative_corr_idxs_dict)

            # Scatter surface mesh across all nodes to preserve computation time and memory.
            gt_scene_trimesh_future = client.scatter(self.loader.gt_scene_trimesh, broadcast=True)

            # Prepare computation graph.
            delayed_sfm_result, delayed_io = self.scene_optimizer.create_computation_graph(
                keypoints_list=keypoints_list,
                putative_corr_idxs_dict=putative_corr_idxs_dict,
                num_images=len(self.loader),
                image_pair_indices=image_pair_indices,
                image_graph=self.loader.create_computation_graph_for_images(),
                all_intrinsics=self.loader.create_computation_graph_for_intrinsics(),
                relative_pose_priors=self.loader.get_relative_pose_priors(image_pair_indices),
                absolute_pose_priors=self.loader.get_absolute_pose_priors(),
                cameras_gt=self.loader.create_computation_graph_for_gt_cameras(),
                gt_wTi_list=self.loader.get_gt_poses(),
                gt_scene_mesh=gt_scene_trimesh_future,
            )

            # Run SfM pipeline.
            sfm_result, *io = dask.compute(delayed_sfm_result, *delayed_io)

        assert isinstance(sfm_result, GtsfmData)
        logger.info("GTSFM took %.2f minutes to compute sparse multi-view result.", (time.time() - start_time) / 60)


if __name__ == "__main__":
    runner = GtsfmRunnerAstrovisionLoader()
    runner.run()
