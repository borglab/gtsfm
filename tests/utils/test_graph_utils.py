"""Unit tests for functions in the graph utils file.

Authors: Ayush Baid
"""
import unittest

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


if __name__ == "__main__":
    unittest.main()
