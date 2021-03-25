"""Unit tests for the multi-view optimizer.

Authors: Ayush Baid
"""
import unittest
import unittest.mock as mock

import numpy as np
from gtsam import EssentialMatrix, Rot3, Unit3

import gtsfm.utils.graph as graph_utils
from gtsfm.multi_view_optimizer import prune_to_largest_connected_component


class TestMultiViewOptimizer(unittest.TestCase):
    """Unit test for the multi-view optimizer."""

    @mock.patch.object(graph_utils, "get_nodes_in_largest_connected_component", return_value=[0, 1, 2, 3])
    def test_prune_to_largest_connected_component(self, graph_largest_cc_mock):
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

        (computed_relative_rotations, computed_relative_unit_translations) = prune_to_largest_connected_component(
            input_relative_rotations, input_relative_unit_translations
        )

        # check the graph util function called with the edges defined by tracks
        graph_largest_cc_mock.assert_called_once_with([(0, 1), (3, 1), (3, 2), (4, 6), (6, 7)])

        expected_edges = [(0, 1), (3, 1), (3, 2)]
        self.assertCountEqual(list(computed_relative_rotations.keys()), expected_edges)
        self.assertCountEqual(list(computed_relative_unit_translations.keys()), expected_edges)

        # check the actual Rot3 and Unit3 values
        for (i1, i2) in expected_edges:
            self.assertTrue(computed_relative_rotations[(i1, i2)].equals(input_relative_rotations[(i1, i2)], 1e-2))
            self.assertTrue(
                computed_relative_unit_translations[(i1, i2)].equals(input_relative_unit_translations[(i1, i2)], 1e-2)
            )


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
    R = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    t = np.random.uniform(low=-1.0, high=1.0, size=(3,))

    return EssentialMatrix(R, Unit3(t))


if __name__ == "__main__":
    unittest.main()
