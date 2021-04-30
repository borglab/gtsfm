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
from hydra.experimental import compose, initialize_config_module
from hydra.utils import instantiate

import gtsfm.utils.geometry_comparisons as comp_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.scene_optimizer import SceneOptimizer

DATA_ROOT_PATH = Path(__file__).resolve().parent / "data"


class TestSceneOptimizer(unittest.TestCase):
    """Unit test for SceneOptimizer, which runs SfM for a scene."""

    def setUp(self) -> None:
        self.loader = OlssonLoader(str(DATA_ROOT_PATH / "set1_lund_door"), image_extension="JPG")
        assert len(self.loader)

    def test_create_computation_graph(self):
        """Will test Dask multi-processing capabilities and ability to serialize all objects."""
        self.loader = OlssonLoader(str(DATA_ROOT_PATH / "set1_lund_door"), image_extension="JPG")

        with initialize_config_module(config_module="gtsfm.configs"):

            # config is relative to the gtsfm module
            cfg = compose(config_name="scene_optimizer_unit_test_config.yaml")
            obj: SceneOptimizer = instantiate(cfg.SceneOptimizer)

            # generate the dask computation graph
            sfm_result_graph = obj.create_computation_graph(
                len(self.loader),
                self.loader.get_valid_pairs(),
                self.loader.create_computation_graph_for_images(),
                self.loader.create_computation_graph_for_intrinsics(),
                self.loader.create_computation_graph_for_image_shapes(),
                gt_pose_graph=self.loader.create_computation_graph_for_poses(),
            )

            # create dask client
            cluster = LocalCluster(n_workers=1, threads_per_worker=4)

            with Client(cluster):
                sfm_result = dask.compute(sfm_result_graph)[0]

            self.assertIsInstance(sfm_result, GtsfmData)

            # compare the camera poses
            computed_poses = sfm_result.get_camera_poses()
            computed_rotations = [x.rotation() for x in computed_poses]
            computed_translations = [x.translation() for x in computed_poses]

            # get active cameras from largest connected component, may be <len(self.loader)
            connected_camera_idxs = sfm_result.get_valid_camera_indices()
            expected_poses = [self.loader.get_camera_pose(i) for i in connected_camera_idxs]

            self.assertTrue(comp_utils.compare_global_poses(expected_poses, expected_poses))


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
    R = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    t = np.random.uniform(low=-1.0, high=1.0, size=(3,))

    return EssentialMatrix(R, Unit3(t))


if __name__ == "__main__":
    unittest.main()
