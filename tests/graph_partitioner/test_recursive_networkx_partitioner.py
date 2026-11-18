"""Tests for recursive NetworkX partitioning (spectral Ncut / ND-style)."""

import unittest

from gtsfm.graph_partitioner.networkx_bisection import bisect_spectral_ncut, bisect_stoer_wagner
from gtsfm.graph_partitioner.recursive_networkx_partitioner import RecursiveNetworkXPartitioner
from gtsfm.products.cluster_tree import ClusterTree


def _leaves_containing(tree: ClusterTree, key: int) -> list[ClusterTree]:
    return [leaf for leaf in tree.leaves() if key in leaf.local_keys()]


class TestRecursiveNetworkXPartitioner(unittest.TestCase):
    """NetworkX recursive cuts, balance, and overlap."""

    def test_empty_input_returns_none(self) -> None:
        self.assertIsNone(RecursiveNetworkXPartitioner().run([]))

    def test_small_graph_is_one_leaf(self) -> None:
        edges = [(0, 1), (1, 2), (0, 2)]
        tree = RecursiveNetworkXPartitioner(max_cameras=5, overlap_cameras=1).run(edges)
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertTrue(tree.is_leaf())
        self.assertEqual(set(tree.value), set(edges))

    def test_binary_cuts_cover_edges_and_share_boundary_cameras(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5), (4, 5), (5, 6), (6, 7)]
        tree = RecursiveNetworkXPartitioner(max_cameras=4, overlap_cameras=2).run(edges)
        self.assertIsNotNone(tree)
        assert tree is not None

        def assert_binary(node: ClusterTree) -> None:
            if node.is_leaf():
                return
            self.assertEqual(len(node.children), 2)
            for child in node.children:
                assert_binary(child)

        assert_binary(tree)
        self.assertEqual(tree.all_edges(), set(edges))
        self.assertTrue(any(len(_leaves_containing(tree, key)) > 1 for key in range(8)))

    def test_spectral_ncut_prefers_light_bridge(self) -> None:
        clique_a = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        clique_b = [(i, j) for i in range(4, 8) for j in range(i + 1, 8)]
        edges = clique_a + clique_b + [(3, 4)]
        weights = {edge: 10 for edge in clique_a + clique_b}
        weights[(3, 4)] = 1
        tree = RecursiveNetworkXPartitioner(
            max_cameras=4, overlap_cameras=0, cutter="nested_dissection", refine_kl=True
        ).run(edges, edge_weights=weights)
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual(len(tree.children), 2)
        sides = sorted((frozenset(child.all_keys()) for child in tree.children), key=min)
        self.assertEqual(sides, [frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})])

    def test_stoer_wagner_balance_rejects_peel(self) -> None:
        clique = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        edges = clique + [(0, 8)]
        adjacency: dict[int, set[int]] = {}
        for i, j in edges:
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)
        keys = tuple(range(9))
        self.assertIsNone(bisect_stoer_wagner(keys, adjacency, max_part_fraction=None))
        parts = bisect_spectral_ncut(keys, adjacency, max_part_fraction=None)
        self.assertIsNotNone(parts)
        assert parts is not None
        sizes = sorted(len(side) for side in parts)
        self.assertEqual(sum(sizes), 9)
        self.assertGreaterEqual(sizes[0], 3)
        self.assertLessEqual(sizes[1], 5)

    def test_nested_dissection_cutter_builds_balanced_tree(self) -> None:
        clique = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        edges = clique + [(0, 8)]
        tree = RecursiveNetworkXPartitioner(max_cameras=5, overlap_cameras=0, cutter="nested_dissection").run(edges)
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertFalse(tree.is_leaf())
        for leaf in tree.leaves():
            self.assertLessEqual(len(leaf.local_keys()), 5)


if __name__ == "__main__":
    unittest.main()
