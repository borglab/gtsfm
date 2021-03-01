"""Unit tests for the scene-optimizer class.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path
from types import SimpleNamespace

import dask
import hydra
import numpy as np
from dask.distributed import LocalCluster, Client
from gtsam import EssentialMatrix, Rot3, Unit3
from hydra.experimental import compose, initialize_config_module
from hydra.utils import instantiate

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.serialization  # import needed to register serialization fns
from gtsfm.common.sfm_result import SfmResult
from gtsfm.loader.folder_loader import FolderLoader
from gtsfm.scene_optimizer import SceneOptimizer
from gtsfm.multi_view_optimizer import select_largest_connected_component

DATA_ROOT_PATH = Path(__file__).resolve().parent / "data"


class TestSceneOptimizer(unittest.TestCase):
    """Unit test for SceneOptimizer, which runs SfM for a scene."""

    def setUp(self) -> None:
        self.loader = FolderLoader(str(DATA_ROOT_PATH / "set1_lund_door"), image_extension="JPG")
        assert len(self.loader)

    def test_find_largest_connected_component(self):
        """Tests the function to prune the scene graph to its largest connected
        component."""

        # create a graph with two connected components of length 4 and 3.
        input_essential_matrices = {
            (0, 1): generate_random_essential_matrix(),
            (1, 5): None,
            (3, 1): generate_random_essential_matrix(),
            (3, 2): generate_random_essential_matrix(),
            (2, 7): None,
            (4, 6): generate_random_essential_matrix(),
            (6, 7): generate_random_essential_matrix(),
        }

        # generate Rot3 and Unit3 inputs
        input_relative_rotations = dict()
        input_relative_unit_translations = dict()
        for (i1, i2), i2Ei1 in input_essential_matrices.items():
            if i2Ei1 is None:
                input_relative_rotations[(i1, i2)] = None
                input_relative_unit_translations[(i1, i2)] = None
            else:
                input_relative_rotations[(i1, i2)] = i2Ei1.rotation()
                input_relative_unit_translations[(i1, i2)] = i2Ei1.direction()

        expected_edges = [(0, 1), (3, 2), (3, 1)]

        (
            computed_relative_rotations,
            computed_relative_unit_translations,
        ) = select_largest_connected_component(input_relative_rotations, input_relative_unit_translations)

        # check the edges in the pruned graph
        self.assertCountEqual(list(computed_relative_rotations.keys()), expected_edges)
        self.assertCountEqual(list(computed_relative_unit_translations.keys()), expected_edges)

        # check the actual Rot3 and Unit3 values
        for (i1, i2) in expected_edges:
            self.assertTrue(computed_relative_rotations[(i1, i2)].equals(input_relative_rotations[(i1, i2)], 1e-2))
            self.assertTrue(
                computed_relative_unit_translations[(i1, i2)].equals(input_relative_unit_translations[(i1, i2)], 1e-2)
            )

    def test_create_computation_graph(self):
        """Will test Dask multi-processing capabilities and ability to serialize all objects."""
        use_intrinsics_in_verification = False

        with initialize_config_module(config_module="gtsfm.configs"):

            # config is relative to the gtsfm module
            cfg = compose(config_name="scene_optimizer_unit_test_config.yaml")
            self.obj: SceneOptimizer = instantiate(cfg.SceneOptimizer)

            # generate the dask computation graph
            sfm_result_graph = self.obj.create_computation_graph(
                len(self.loader),
                self.loader.get_valid_pairs(),
                self.loader.create_computation_graph_for_images(),
                self.loader.create_computation_graph_for_intrinsics(),
                use_intrinsics_in_verification=use_intrinsics_in_verification,
            )

            # create dask client
            cluster = LocalCluster(n_workers=1, threads_per_worker=4)

            with Client(cluster):
                sfm_result = dask.compute(sfm_result_graph)[0]

            self.assertIsInstance(sfm_result, SfmResult)

            # compare the camera poses
            computed_poses = sfm_result.get_camera_poses()
            computed_rotations = [x.rotation() for x in computed_poses]
            computed_translations = [x.translation() for x in computed_poses]

            expected_poses = [self.loader.get_camera_pose(i) for i in range(len(self.loader))]
            expected_rotations = [x.rotation() for x in expected_poses]
            expected_translations = [x.translation() for x in expected_poses]

            self.assertTrue(comp_utils.align_and_compare_rotations(computed_rotations, expected_rotations, 2))
            self.assertTrue(
                comp_utils.align_and_compare_translations(computed_translations, expected_translations, 1e-1, 1e-1)
            )


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
    R = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    t = np.random.uniform(low=-1.0, high=1.0, size=(3,))

    return EssentialMatrix(R, Unit3(t))


if __name__ == "__main__":
    unittest.main()
