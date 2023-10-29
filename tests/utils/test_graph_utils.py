"""Unit tests for functions in the graph utils file.

Authors: Ayush Baid, John Lambert, Akshay Krishnan
"""
import unittest
from collections import defaultdict
from types import SimpleNamespace
from typing import List, Tuple
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
        ) = graph_utils.prune_to_largest_connected_component(
            input_relative_rotations, input_relative_unit_translations, relative_pose_priors={}
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

    def test_extract_quadruplets_0found_0edges(self) -> None:
        edges = []
        quadruplets = graph_utils.extract_cyclic_quadruplets_from_edges(edges)
        assert isinstance(quadruplets, list)
        assert len(quadruplets) == 0


    def test_extract_quadruplets_0found_8edges(self) -> None:
        """
              2 --------- 7
           /  |           |
         /    |           |
        1 --- 0 --- 4 --- 5
              |
              |
              3 --- 6
        """
        edges = [
            (1,2),
            (2,7),
            (5,7),
            (4,5),
            (0,4),
            (0,2),
            (0,3),
            (3,6)
        ]
        quadruplets = graph_utils.extract_cyclic_quadruplets_from_edges(edges)
        assert isinstance(quadruplets, list)
        assert len(quadruplets) == 0

    def test_extract_quadruplets_1found(self) -> None:
        """
        0 --- 1
        |     |
        |     |
        3 --- 2 --- 4
        """
        edges = [
            (0,1),
            (1,2),
            (2,3),
            (3,0),
            (2,4)
        ]
        quadruplets = graph_utils.extract_cyclic_quadruplets_from_edges(edges)

        assert isinstance(quadruplets, list)
        assert len(quadruplets) == 1
        assert quadruplets[0] == (0, 1, 2, 3)


    def test_extract_quadruplets_2found(self) -> None:
        """
        0 ---- 1
        |      |\
        |      | \
        |      |  \
        2 ---- 3   \
               |    \
               |     \
               |      \
               4 ----- 5
        """
        edges = [
            (0,1),
            (1,3),
            (2,3),
            (0,2),
            (1,5),
            (3,4),
            (4,5)
        ]
        quadruplets = graph_utils.extract_cyclic_quadruplets_from_edges(edges)
        
        assert isinstance(quadruplets, list)
        assert len(quadruplets) == 2
        assert quadruplets[0] == (1, 3, 4, 5)
        assert quadruplets[1] == (0, 1, 3, 2) 


    def test_extract_triplets_1(self) -> None:
        """Ensure triplets are recovered accurately via intersection of adjacency lists.

        Consider the following undirected graph with 1 cycle:

        0 ---- 1
              /|
             / |
            /  |
          2 -- 3
               |
               |
               4
        """
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (1, 3),
            (3, 4),
        ]
        triplets = graph_utils.extract_cyclic_triplets_from_edges(edges)
        assert isinstance(triplets, list)
        assert len(triplets) == 1
        assert triplets[0] == (1, 2, 3)

    def test_extract_triplets_2(self) -> None:
        """Ensure triplets are recovered accurately via intersection of adjacency lists.

        Consider the following undirected graph with 2 cycles. The cycles share an edge:

        0 ---- 1
              /|\
             / | \
            /  |  \
          2 -- 3 -- 5
               |
               |
               4
        """
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (1, 3),
            (3, 4),
            (1, 5),
            (3, 5),
        ]
        triplets = graph_utils.extract_cyclic_triplets_from_edges(edges)
        assert isinstance(triplets, list)
        assert len(triplets) == 2
        self.assertIn((1, 2, 3), triplets)
        self.assertIn((1, 3, 5), triplets)

    def test_extract_triplets_3(self) -> None:
        """Ensure triplets are recovered accurately via intersection of adjacency lists.

        Consider the following undirected graph with 2 cycles. The cycles share a node:

        0 ---- 1
              /|
             / |
            /  |
          2 -- 3
               |\
               | \
               |  \
               4 -- 5
        """
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (1, 3),
            (3, 4),
            (3, 5),
            (4, 5),
        ]
        triplets = graph_utils.extract_cyclic_triplets_from_edges(edges)
        assert isinstance(triplets, list)
        assert len(triplets) == 2
        assert triplets[0] == (3, 4, 5)
        assert triplets[1] == (1, 2, 3)

    def test_triplet_extraction_correctness(self) -> None:
        """Ensure that for large graphs, the adjacency-list-based algorithm is still correct,
        when compared with the brute-force O(n^3) implementation.
        """
        num_pairs = 100
        # suppose we have 200 images for a scene
        pairs = np.random.randint(low=0, high=200, size=(num_pairs, 2))
        # i1 < i2 by construction inside loader classes
        pairs = np.sort(pairs, axis=1)

        # remove edges that would represent self-loops, i.e. (i1,i1) is not valid for a measurement
        invalid = pairs[:, 0] == pairs[:, 1]
        pairs = pairs[~invalid]
        edges = pairs.tolist()
        triplets = graph_utils.extract_cyclic_triplets_from_edges(edges)

        # Now, compare with the brute force method
        triplets_bf = extract_triplets_brute_force(edges)

        assert set(triplets) == set(triplets_bf)

    def test_create_adjacency_list(self) -> None:
        """Ensure the generated adjacency graph is empty, for a simple graph.

        Graph topology (assume all graph vertices have the same orientation):

        0 ---- 1
              /|
             / |
            /  |
          2 -- 3
               |\
               | \
               |  \
               4 -- 5
        """
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (1, 3),
            (3, 4),
            (4, 5),
            (3, 5),
        ]
        adj_list = graph_utils.create_adjacency_list(edges)

        # fmt: off
        # expected_adj_list = {
        #     0: {1},
        #     1: {0, 2, 3},
        #     2: {1, 3},
        #     3: {1, 2, 4, 5},
        #     4: {3, 5},
        #     5: {3, 4}
        # }
        # fmt: on
        assert isinstance(adj_list, defaultdict)

    def test_create_adjacency_list_empty(self) -> None:
        """Ensure the generated adjacency graph is empty, when no edges are provided."""
        edges = []
        adj_list = graph_utils.create_adjacency_list(edges)

        assert len(adj_list.keys()) == 0
        assert isinstance(adj_list, defaultdict)

    def test_draw_view_graph_topology(self) -> None:
        """Make sure we can draw a simple graph topology using networkx."""
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        two_view_reports_w_gt_errors = {
            (0, 1): SimpleNamespace(**{"R_error_deg": 1, "U_error_deg": 1}),
            (1, 2): SimpleNamespace(**{"R_error_deg": 1, "U_error_deg": 1}),
            (2, 3): SimpleNamespace(**{"R_error_deg": 1, "U_error_deg": 1}),
            (0, 3): SimpleNamespace(**{"R_error_deg": 10, "U_error_deg": 0}),
        }
        title = "dummy_4_image_cycle"
        save_fpath = "plot.jpg"
        graph_utils.draw_view_graph_topology(
            edges=edges,
            two_view_reports=two_view_reports_w_gt_errors,
            title=title,
            save_fpath=save_fpath,
            cameras_gt=None,
        )

        two_view_reports = {
            (0, 1): SimpleNamespace(**{"R_error_deg": None, "U_error_deg": None}),
            (1, 2): SimpleNamespace(**{"R_error_deg": None, "U_error_deg": None}),
            (2, 3): SimpleNamespace(**{"R_error_deg": None, "U_error_deg": None}),
            (0, 3): SimpleNamespace(**{"R_error_deg": None, "U_error_deg": None}),
        }
        graph_utils.draw_view_graph_topology(
            edges=edges, two_view_reports=two_view_reports, title=title, save_fpath=save_fpath, cameras_gt=None
        )


def extract_triplets_brute_force(edges: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """Use triple for-loop to find triplets from a graph G=(V,E) in O(n^3) time.

    Note: this method should **never** be used in practice, other than for exhaustively checking for correctness.
    It is a **much** slower implementation for large graphs, when compared to `extract_triplets()` that uses
    intersection of adjacency lists. It is used to check correctness inside the unit test below.

    Args:
        edges: edges between image pair indices.

    Returns:
        triplets: 3-tuples of nodes that form a cycle. Nodes of each triplet are provided in sorted order.
    """
    triplets = set()

    for (i1, i2) in edges:
        for (j1, j2) in edges:
            for (k1, k2) in edges:
                # check how many nodes are spanned by these 3 edges
                cycle_nodes = set([i1, i2]).union(set([j1, j2])).union(set([k1, k2]))
                # sort them in increasing order
                cycle_nodes = tuple(sorted(cycle_nodes))

                # nodes cannot be repeated
                unique_edges = set([(i1, i2), (j1, j2), (k1, k2)])
                edges_are_unique = len(unique_edges) == 3

                if len(cycle_nodes) == 3 and edges_are_unique:
                    triplets.add(cycle_nodes)
    return list(triplets)


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
    R = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    t = np.random.uniform(low=-1.0, high=1.0, size=(3,))

    return EssentialMatrix(R, Unit3(t))


if __name__ == "__main__":
    unittest.main()
