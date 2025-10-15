"""Unit tests for the MetisPartitioner."""

import unittest

from gtsam import SymbolicBayesTreeClique  # type: ignore

from gtsfm.graph_partitioner.metis_partitioner import MetisPartitioner
from gtsfm.products.cluster_tree import Cluster


class TestMetisPartitioner(unittest.TestCase):
    """Tests for the METIS-based graph partitioner."""

    def setUp(self) -> None:
        self.chain_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
        # Skydio-32 visibility graph (integers only).
        # fmt: off
        self.skydio_pairs = [
            (7, 17), (15, 30), (4, 9), (3, 13), (8, 18), (0, 14), (11, 23), (7, 10), (15, 23), (18, 19), (3, 15), (22, 28), (8, 11), (14, 24), (11, 16), (15, 25), (7, 21), (3, 8), (14, 17), (0, 9), (15, 18), (3, 10), (0, 2), (14, 19), (14, 28), (10, 24), (7, 16), (29, 31), (3, 12), (22, 25), (21, 27), (10, 17), (2, 13), (3, 14), (14, 23), (2, 6), (17, 25), (7, 11), (6, 15), (3, 7), (3, 16), (14, 16), (14, 25), (10, 12), (2, 8), (10, 30), (25, 28), (6, 17), (3, 9), (14, 18), (17, 20), (10, 14), (10, 23), (2, 10), (13, 25), (6, 19), (17, 22), (9, 18), (17, 31), (10, 25), (13, 27), (3, 4), (6, 21), (9, 11), (17, 24), (10, 18), (2, 14), (9, 29), (13, 20), (6, 14), (24, 29), (16, 25), (5, 15), (2, 7), (9, 22), (6, 7), (13, 22), (6, 16), (16, 18), (24, 31), (16, 27), (5, 8), (5, 17), (17, 19), (9, 15), (2, 9), (13, 15), (6, 9), (13, 24), (6, 18), (16, 20), (5, 10), (23, 25), (5, 19), (9, 17), (17, 30), (1, 15), (13, 17), (6, 11), (16, 22), (12, 18), (5, 12), (2, 4), (17, 23), (23, 27), (13, 28), (5, 14), (23, 29), (9, 12), (1, 10), (27, 29), (13, 21), (16, 17), (24, 30), (5, 7), (12, 22), (5, 16), (4, 18), (12, 31), (23, 31), (1, 3), (27, 31), (1, 12), (16, 19), (5, 9), (23, 24), (5, 18), (9, 16), (1, 5), (8, 20), (13, 16), (19, 29), (16, 21), (16, 30), (12, 17), (5, 11), (8, 13), (1, 7), (19, 22), (1, 16), (16, 23), (15, 27), (4, 6), (12, 19), (5, 13), (4, 15), (23, 28), (8, 15), (27, 28), (1, 9), (0, 11), (19, 24), (18, 25), (4, 8), (1, 2), (0, 4), (8, 17), (0, 13), (11, 22), (11, 31), (15, 22), (7, 18), (4, 10), (12, 23), (4, 19), (1, 4), (0, 6), (0, 15), (11, 15), (19, 28), (11, 24), (7, 20), (18, 20), (12, 16), (18, 29), (4, 12), (8, 12), (1, 6), (0, 8), (8, 21), (19, 21), (19, 30), (15, 17), (7, 13), (7, 22), (18, 22), (4, 14), (18, 31), (0, 1), (22, 31), (8, 14), (14, 27), (0, 10), (11, 19), (15, 19), (7, 15), (15, 28), (26, 28), (25, 30), (4, 7), (4, 16), (22, 24), (0, 3), (14, 20), (11, 21),  # noqa: E501
        ]
        # fmt: on

    def test_empty_input_returns_empty_clustering(self) -> None:
        partitioner = MetisPartitioner()
        cluster_tree = partitioner.run([])
        self.assertTrue(cluster_tree.is_empty())
        self.assertEqual(cluster_tree.leaves(), ())

    def test_chain_graph_creates_non_empty_clustering(self) -> None:
        partitioner = MetisPartitioner()
        cluster_tree = partitioner.run(self.chain_edges)
        self.assertFalse(cluster_tree.is_empty())
        leaves = cluster_tree.leaves()
        self.assertGreater(len(leaves), 0)
        for leaf in leaves:
            self.assertIsInstance(leaf, Cluster)
            self.assertTrue(all(i < j for i, j in leaf.edges))

    def test_group_by_leaf_matches_edges(self) -> None:
        partitioner = MetisPartitioner()
        cluster_tree = partitioner.run(self.chain_edges)
        grouped = cluster_tree.group_by_leaf({edge: edge for edge in self.chain_edges})
        self.assertEqual(len(grouped), len(cluster_tree.leaves()))

    def test_clique_key_sets(self) -> None:
        partitioner = MetisPartitioner()
        bayes_tree = partitioner.symbolic_bayes_tree(self.skydio_pairs)
        root: SymbolicBayesTreeClique = bayes_tree.roots()[0]
        frontals, separator = partitioner._clique_key_sets(root)
        self.assertEqual(frontals, {1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 25, 27, 28})
        self.assertEqual(separator, set())
        self.assertTrue(root.nrChildren() > 0)
        clique = root[0]
        frontals2, separator2 = partitioner._clique_key_sets(clique)
        self.assertEqual(frontals2, {0})
        self.assertEqual(separator2, {1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})

    def test_skydio_leaf_edges_are_intra_cluster(self) -> None:
        partitioner = MetisPartitioner()
        cluster_tree = partitioner.run(self.skydio_pairs)
        assert cluster_tree.root is not None
        self.assertEqual(len(cluster_tree.root.all_edges()), len(self.skydio_pairs))
        self.assertFalse(cluster_tree.is_empty())
        for cluster in cluster_tree.leaves():
            # All edge endpoints must lie inside the cluster key set.
            leaf_keys = cluster.all_keys()
            for i, j in cluster.edges:
                self.assertIn(i, leaf_keys)
                self.assertIn(j, leaf_keys)
            if len(leaf_keys) <= 1:
                self.assertEqual(len(cluster.edges), 0)


if __name__ == "__main__":
    unittest.main()
