"""Unit tests for the scene-optimizer class.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

import dask
import hydra
import numpy as np
from dask.distributed import LocalCluster, Client
from gtsam import EssentialMatrix, Rot3, Unit3
from hydra.utils import instantiate

import gtsfm.utils.geometry_comparisons as comp_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.scene_optimizer import SceneOptimizer

ROOT_PATH = Path(__file__).resolve().parent.parent


class TestSceneOptimizer(unittest.TestCase):
    """Unit test for SceneOptimizer, which runs SfM for a scene."""

    def test_create_computation_graph(self):
        """Will test Dask multi-processing capabilities and ability to serialize all objects."""

        with hydra.initialize(config_path="../gtsfm/configs"):

            # config is relative to the gtsfm module
            cfg = hydra.compose(
                config_name="defaults",
                overrides=[
                    "loader.folder={}".format(ROOT_PATH / "tests" / "data" / "set1_lund_door"),
                    "loader.image_extension=JPG",
                ],
            )
            obj: SceneOptimizer = instantiate(cfg.scene_optimizer)
            loader: LoaderBase = instantiate(cfg.loader)

            # generate the dask computation graph
            sfm_result_graph = obj.create_computation_graph(
                num_images=len(loader),
                image_pair_indices=loader.get_valid_pairs(),
                image_graph=loader.create_computation_graph_for_images(),
                camera_intrinsics_graph=loader.create_computation_graph_for_intrinsics(),
                image_shape_graph=loader.create_computation_graph_for_image_shapes(),
                gt_pose_graph=loader.create_computation_graph_for_poses(),
            )
            # create dask client
            cluster = LocalCluster(n_workers=cfg.dask.num_workers, threads_per_worker=cfg.dask.threads_per_worker)

            with Client(cluster):
                sfm_result = dask.compute(sfm_result_graph)[0]

            self.assertIsInstance(sfm_result, GtsfmData)

            # compare the camera poses
            computed_poses = sfm_result.get_camera_poses()

            # get active cameras from largest connected component, may be <len(self.loader)
            connected_camera_idxs = sfm_result.get_valid_camera_indices()
            expected_poses = [self.loader.get_camera_pose(i) for i in connected_camera_idxs]

            self.assertTrue(
                comp_utils.compare_global_poses(expected_poses, computed_poses, trans_err_atol=1.00, trans_err_rtol=0.1)
            )


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
    R = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    t = np.random.uniform(low=-1.0, high=1.0, size=(3,))

    return EssentialMatrix(R, Unit3(t))


if __name__ == "__main__":
    unittest.main()
