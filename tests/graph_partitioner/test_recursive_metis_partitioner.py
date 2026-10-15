"""Tests for recursive two-way METIS partitioning."""

import unittest

from gtsfm.graph_partitioner.recursive_metis_partitioner import RecursiveMetisPartitioner
from gtsfm.products.cluster_tree import ClusterTree


def _leaves_containing(tree: ClusterTree, key: int) -> list[ClusterTree]:
    return [leaf for leaf in tree.leaves() if key in leaf.local_keys()]


class TestRecursiveMetisPartitioner(unittest.TestCase):
    """Recursive METIS cuts and boundary-camera overlap."""

    def test_empty_input_returns_none(self) -> None:
        self.assertIsNone(RecursiveMetisPartitioner().run([]))

    def test_small_graph_is_one_leaf(self) -> None:
        edges = [(0, 1), (1, 2), (0, 2)]
        tree = RecursiveMetisPartitioner(max_cameras=5, overlap_cameras=1).run(edges)
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertTrue(tree.is_leaf())
        self.assertEqual(set(tree.value), set(edges))

    def test_binary_cuts_cover_edges_and_share_boundary_cameras(self) -> None:
        # Facade 0-1-2-3 and courtyard 4-5-6-7, joined by 1-4 and 2-5.
        edges = [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5), (4, 5), (5, 6), (6, 7)]
        tree = RecursiveMetisPartitioner(max_cameras=4, overlap_cameras=2).run(edges)
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

        facade_leaf = _leaves_containing(tree, 0)[0]
        courtyard_leaf = _leaves_containing(tree, 7)[0]
        self.assertIn(4, facade_leaf.local_keys())
        self.assertIn(1, courtyard_leaf.local_keys())
        self.assertIn(5, _leaves_containing(tree, 3)[0].local_keys())
        self.assertIn(2, _leaves_containing(tree, 4)[0].local_keys())
        for leaf in tree.leaves():
            self.assertTrue(all(i < j for i, j in leaf.value))

    def test_relaxed_balance_can_cut_off_a_small_side(self) -> None:
        # Two cliques joined by one bridge. A 50/50 cut must sever a clique.
        clique_a = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        edges = clique_a + [(8, 9), (0, 8)]
        tree = RecursiveMetisPartitioner(max_cameras=9, overlap_cameras=0, max_part_fraction=0.8).run(edges)
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual(len(tree.children), 2)
        sizes = sorted(len(child.all_keys()) for child in tree.children)
        self.assertEqual(sizes, [2, 8])


if __name__ == "__main__":
    unittest.main()
