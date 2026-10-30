"""Tests for recursive NetworkX weighted min-cut partitioning."""

import unittest

from gtsfm.graph_partitioner.recursive_weighted_mincut_partitioner import RecursiveWeightedMinCutPartitioner
from gtsfm.products.cluster_tree import ClusterTree


def _leaves_containing(tree: ClusterTree, key: int) -> list[ClusterTree]:
    return [leaf for leaf in tree.leaves() if key in leaf.local_keys()]


class TestRecursiveWeightedMinCutPartitioner(unittest.TestCase):
    """Recursive weighted min-cut and boundary-camera overlap."""

    def test_empty_input_returns_none(self) -> None:
        self.assertIsNone(RecursiveWeightedMinCutPartitioner().run([]))

    def test_small_graph_is_one_leaf(self) -> None:
        edges = [(0, 1), (1, 2), (0, 2)]
        tree = RecursiveWeightedMinCutPartitioner(max_cameras=5, overlap_cameras=1).run(edges)
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertTrue(tree.is_leaf())
        self.assertEqual(set(tree.value), set(edges))

    def test_binary_cuts_cover_edges_and_share_boundary_cameras(self) -> None:
        # Facade 0-1-2-3 and courtyard 4-5-6-7, joined by 1-4 and 2-5.
        edges = [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5), (4, 5), (5, 6), (6, 7)]
        tree = RecursiveWeightedMinCutPartitioner(max_cameras=4, overlap_cameras=2).run(edges)
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
        # Overlap copies boundary cameras into at least one leaf on the far side of a cut.
        self.assertTrue(any(len(_leaves_containing(tree, key)) > 1 for key in range(8)))
        for leaf in tree.leaves():
            self.assertTrue(all(i < j for i, j in leaf.value))

    def test_overlap_participates_in_later_cuts_and_respects_cap(self) -> None:
        clique_a = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        clique_b = [(i, j) for i in range(4, 8) for j in range(i + 1, 8)]
        edges = clique_a + clique_b + [(3, 4)]
        capped = RecursiveWeightedMinCutPartitioner(
            max_cameras=4, overlap_cameras=2, overlap_participates_in_cuts=True
        ).run(edges)
        self.assertIsNotNone(capped)
        assert capped is not None
        self.assertTrue(all(len(leaf.local_keys()) <= 4 for leaf in capped.leaves()))
        self.assertGreater(len(capped.leaves()), 2)
        uncapped = RecursiveWeightedMinCutPartitioner(max_cameras=4, overlap_cameras=2).run(edges)
        assert uncapped is not None
        self.assertTrue(any(len(leaf.local_keys()) > 4 for leaf in uncapped.leaves()))

    def test_light_edge_is_preferred_cut(self) -> None:
        # Path 0-1-2-3 with a light middle edge: weighted min-cut separates {0,1} | {2,3}.
        edges = [(0, 1), (1, 2), (2, 3)]
        weights = {(0, 1): 10, (1, 2): 1, (2, 3): 10}
        tree = RecursiveWeightedMinCutPartitioner(max_cameras=2, overlap_cameras=0).run(edges, edge_weights=weights)
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual(len(tree.children), 2)
        self.assertEqual(set(tree.value), {(1, 2)})
        sides = sorted((frozenset(child.all_keys()) for child in tree.children), key=min)
        self.assertEqual(sides, [frozenset({0, 1}), frozenset({2, 3})])

    def test_balance_guard_rejects_extreme_peels(self) -> None:
        # A pendant node is the true min-cut; with max_part_fraction=0.8 that peel is rejected
        # and the midpoint fallback yields a more balanced split.
        clique = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        edges = clique + [(0, 8)]
        tree = RecursiveWeightedMinCutPartitioner(max_cameras=8, overlap_cameras=0, max_part_fraction=0.8).run(edges)
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual(len(tree.children), 2)
        sizes = sorted(len(child.all_keys()) for child in tree.children)
        self.assertGreaterEqual(sizes[0], 2)
        self.assertLessEqual(sizes[1], 7)


if __name__ == "__main__":
    unittest.main()
