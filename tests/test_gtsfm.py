"""Unit tests for the the main driver class.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

import dask
import numpy as np
from gtsam import EssentialMatrix, Rot3, Unit3

from frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from frontend.matcher.twoway_matcher import TwoWayMatcher
from frontend.verifier.degensac import Degensac
from gtsfm import GTSFM
from loader.folder_loader import FolderLoader


DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / 'data'


class TestGTSFM(unittest.TestCase):
    """[summary]

    Args:
        unittest ([type]): [description]
    """

    def setUp(self) -> None:
        self.loader = FolderLoader(
            str(DATA_ROOT_PATH / 'set1_lund_door'), image_extension='JPG')
        assert len(self.loader)
        self.obj = GTSFM(
            detector_descriptor=SIFTDetectorDescriptor(),
            matcher=TwoWayMatcher(),
            verifier=Degensac()
        )

    def test_find_largest_connected_component(self):
        """Tests the function to prune the scene graph to its largest connected
        component."""

        # create a graph with two connected components of length 4 and 3.
        input_essential_matrices = {
            (0, 1): generate_random_essential_matrix(),
            (1, 5): None,
            (1, 3): generate_random_essential_matrix(),
            (3, 2): generate_random_essential_matrix(),
            (2, 7): None,
            (4, 6): generate_random_essential_matrix(),
            (6, 7): generate_random_essential_matrix()
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

        expected_edges = [(0, 1), (3, 2), (1, 3)]

        computed_relative_rotations, computed_relative_unit_translations = \
            GTSFM.select_largest_connected_component(
                input_relative_rotations, input_relative_unit_translations)

        # check the edges in the pruned graph
        self.assertCountEqual(
            list(computed_relative_rotations.keys()), expected_edges)
        self.assertCountEqual(
            list(computed_relative_unit_translations.keys()), expected_edges)

        # check the actual Rot3 and Unit3 values
        for (i1, i2) in expected_edges:
            self.assertTrue(
                computed_relative_rotations[(i1, i2)].equals(
                    input_relative_rotations[(i1, i2)], 1e-2))
            self.assertTrue(
                computed_relative_unit_translations[(i1, i2)].equals(
                    input_relative_unit_translations[(i1, i2)], 1e-2))

    def test_create_computation_graph(self):

        exact_intrinsics_flag = False

        # run normally without dask
        expected_keypoints_list, \
            expected_relative_rotations, \
            expected_relative_unit_translations, \
            expected_verified_corr_indices = self.obj.run(
                self.loader, exact_intrinsics_flag=exact_intrinsics_flag)

        # generate the dask computation graph
        keypoints_graph, relative_rotations_graph, relative_translations_graph, verified_corr_graph = self.obj.create_computation_graph(
            self.loader.get_valid_pairs(),
            self.loader.create_computation_graph_for_images(),
            self.loader.create_computation_graph_for_intrinsics(),
            exact_intrinsics_flag=exact_intrinsics_flag
        )

        with dask.config.set(scheduler='single-threaded'):
            computed_keypoints_list = dask.compute(keypoints_graph)[0]
            computed_relative_rotations = dask.compute(
                relative_rotations_graph)[0]
            computed_relative_unit_translations = \
                dask.compute(relative_translations_graph)[0]
            computed_verified_corr_indices = \
                dask.compute(verified_corr_graph)[0]

        # compute the number of length of lists and dictionaries
        self.assertEqual(len(computed_keypoints_list),
                         len(expected_keypoints_list))
        self.assertEqual(len(computed_relative_rotations),
                         len(expected_relative_rotations))
        self.assertEqual(len(computed_relative_unit_translations),
                         len(expected_relative_unit_translations))
        self.assertEqual(len(computed_verified_corr_indices),
                         len(expected_verified_corr_indices))

        # compare keypoints for all indices
        self.assertListEqual(computed_keypoints_list, expected_keypoints_list)


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(
        low=0.0, high=2*np.pi, size=(3,))
    R = Rot3.RzRyRx(
        rotation_angles[0], rotation_angles[1], rotation_angles[2])
    T = np.random.uniform(
        low=-1.0, high=1.0, size=(3, ))

    return EssentialMatrix(R, Unit3(T))


if __name__ == "__main__":
    unittest.main()
