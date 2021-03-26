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
from gtsfm.common.sfm_result import SfmResult
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

        use_intrinsics_in_verification = False

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
                use_intrinsics_in_verification=use_intrinsics_in_verification,
                gt_pose_graph=self.loader.create_computation_graph_for_poses(),
            )

            # create dask client
            cluster = LocalCluster(n_workers=1, threads_per_worker=4)

            with Client(cluster):
                sfm_result = dask.compute(sfm_result_graph)[0]

            self.assertIsInstance(sfm_result, SfmResult)

            # compare the camera poses
            poses = sfm_result.get_camera_poses()

            expected_poses = [self.loader.get_camera_pose(i) for i in range(len(self.loader))]

            self.assertTrue(
                comp_utils.compare_global_poses(poses, expected_poses, rot_err_thresh=0.03, trans_err_thresh=0.35)
            )


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
    R = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    t = np.random.uniform(low=-1.0, high=1.0, size=(3,))

    return EssentialMatrix(R, Unit3(t))


if __name__ == "__main__":
    unittest.main()
