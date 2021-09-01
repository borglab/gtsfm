"""Unit tests for functions in the graph utils file.

Authors: Ayush Baid
"""
import unittest
from unittest import mock

import numpy as np
from gtsam import EssentialMatrix, Rot3, Unit3

import gtsfm.utils.graph as graph_utils


class TestGraphUtils(unittest.TestCase):
    def test_get_nodes_in_largest_connected_component_two_components(self):
        """Testing the function to get nodes in largest connected component with an input graph of two connected
        components."""
        edges_component_1 = [(2, 4), (3, 4), (4, 7), (7, 6)]
        edges_component_2 = [(1, 5), (8, 9)]
        edges_all = edges_component_1 + edges_component_2

        computed = graph_utils.get_nodes_in_largest_connected_component(edges_all)
        expected = [2, 3, 4, 6, 7]

        self.assertListEqual(sorted(computed), sorted(expected))

    def test_get_nodes_in_largest_connected_component_no_edges(self):
        """Testing the function to get nodes in largest connected component with an input of empty graph."""
        computed = graph_utils.get_nodes_in_largest_connected_component([])
        self.assertListEqual(computed, [])

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

        (
            computed_relative_rotations,
            computed_relative_unit_translations,
        ) = graph_utils.prune_to_largest_connected_component(input_relative_rotations, input_relative_unit_translations)

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
