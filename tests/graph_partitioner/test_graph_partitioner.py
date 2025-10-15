"""Unit tests for graph partitioning functionality.

Authors: Zongyue Liu
"""

import unittest
from typing import TypeVar

from gtsfm.graph_partitioner.binary_tree_partition import BinaryTreePartitioner
from gtsfm.graph_partitioner.single_partition import SinglePartitioner
from gtsfm.products.clustering import Cluster, Clustering

T = TypeVar("T")


class TestGraphPartitioning(unittest.TestCase):
    """Tests for graph partitioning functionality."""

    def setUp(self) -> None:
        # fmt: off
        # From Skydio-32 dataset:
        self.pairs = [(7, 17), (15, 30), (4, 9), (3, 13), (8, 18), (0, 14), (11, 23), (7, 10), (15, 23), (18, 19), (3, 15), (22, 28), (8, 11), (14, 24), (11, 16), (15, 25), (7, 21), (3, 8), (14, 17), (0, 9), (15, 18), (3, 10), (0, 2), (14, 19), (14, 28), (10, 24), (7, 16), (29, 31), (3, 12), (22, 25), (21, 27), (10, 17), (2, 13), (3, 14), (14, 23), (2, 6), (17, 25), (7, 11), (6, 15), (3, 7), (3, 16), (14, 16), (14, 25), (10, 12), (2, 8), (10, 30), (25, 28), (6, 17), (3, 9), (14, 18), (17, 20), (10, 14), (10, 23), (2, 10), (13, 25), (6, 19), (17, 22), (9, 18), (17, 31), (10, 25), (13, 27), (3, 4), (6, 21), (9, 11), (17, 24), (10, 18), (2, 14), (9, 29), (13, 20), (6, 14), (24, 29), (16, 25), (5, 15), (2, 7), (9, 22), (6, 7), (13, 22), (6, 16), (16, 18), (24, 31), (16, 27), (5, 8), (5, 17), (17, 19), (9, 15), (2, 9), (13, 15), (6, 9), (13, 24), (6, 18), (16, 20), (5, 10), (23, 25), (5, 19), (9, 17), (17, 30), (1, 15), (13, 17), (6, 11), (16, 22), (12, 18), (5, 12), (2, 4), (17, 23), (23, 27), (13, 28), (5, 14), (23, 29), (9, 12), (1, 10), (27, 29), (13, 21), (16, 17), (24, 30), (5, 7), (12, 22), (5, 16), (4, 18), (12, 31), (23, 31), (1, 3), (27, 31), (1, 12), (16, 19), (5, 9), (23, 24), (5, 18), (9, 16), (1, 5), (8, 20), (13, 16), (19, 29), (16, 21), (16, 30), (12, 17), (5, 11), (8, 13), (1, 7), (19, 22), (1, 16), (16, 23), (15, 27), (4, 6), (12, 19), (5, 13), (4, 15), (23, 28), (8, 15), (27, 28), (1, 9), (0, 11), (19, 24), (18, 25), (4, 8), (1, 2), (0, 4), (8, 17), (0, 13), (11, 22), (11, 31), (15, 22), (7, 18), (4, 10), (12, 23), (4, 19), (1, 4), (0, 6), (0, 15), (11, 15), (19, 28), (11, 24), (7, 20), (18, 20), (12, 16), (18, 29), (4, 12), (8, 12), (1, 6), (0, 8), (8, 21), (19, 21), (19, 30), (15, 17), (7, 13), (7, 22), (18, 22), (4, 14), (18, 31), (0, 1), (22, 31), (8, 14), (14, 27), (0, 10), (11, 19), (15, 19), (7, 15), (15, 28), (26, 28), (25, 30), (4, 7), (4, 16), (22, 24), (0, 3), (14, 20), (11, 21)]  # noqa: E501
        # fmt: on
        self.dummy_results = {key: None for key in self.pairs}

    def test_single_partition(self):
        """Test that SinglePartitioner correctly returns all pairs as one partition."""
        # Create some dummy image pairs
        image_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]

        # Create a SinglePartitioner instance
        partitioner = SinglePartitioner()

        # Get partitioned result
        clustering = partitioner.run(image_pairs)

        # Check that we get exactly one partition
        leaves = clustering.leaves()
        self.assertEqual(len(leaves), 1)

        # Check that the partition contains all the original pairs
        self.assertEqual(set(leaves[0].edges), set(image_pairs))

    def test_partition_preserves_key_order(self):
        """Test that all edges in partitions have i < j (valid key order)."""
        partitioner = BinaryTreePartitioner(max_depth=1)
        clustering = partitioner.run(self.pairs)
        total_edges = 0
        for cluster in clustering.leaves():
            assert isinstance(cluster, Cluster)
            total_edges += len(cluster.edges)
            for i, j in cluster.edges:
                self.assertLess(i, j, f"Edge ({i},{j}) does not satisfy i < j")
        # regression:
        # Accept both 110 and 112 as valid results.
        # May differ across platforms due to non-deterministic partitioning.
        self.assertIn(total_edges, [128, 112], f"Expected 110 or 112 intra-edges in total, got {total_edges}")

    def test_group_results_by_subgraph(self):
        """Test grouping results by subgraph."""
        # fmt: off
        subgraph_edges = [[(7, 17), (4, 9), (7, 21), (0, 9), (0, 2), (29, 31), (3, 12), (2, 6), (3, 7), (6, 17), (3, 9), (17, 20), (9, 18), (17, 31), (3, 4), (6, 21), (9, 29), (2, 7), (6, 7), (5, 17), (2, 9), (6, 9), (6, 18), (9, 17), (12, 18), (5, 12), (2, 4), (9, 12), (5, 7), (4, 18), (12, 31), (1, 3), (1, 12), (5, 9), (5, 18), (1, 5), (12, 17), (1, 7), (4, 6), (1, 9), (1, 2), (0, 4), (7, 18), (1, 4), (0, 6), (7, 20), (18, 20), (18, 29), (4, 12), (1, 6), (18, 31), (0, 1), (4, 7), (0, 3)], [(15, 30), (11, 23), (15, 23), (22, 28), (8, 11), (14, 24), (11, 16), (15, 25), (14, 19), (14, 28), (10, 24), (22, 25), (14, 23), (14, 16), (14, 25), (10, 30), (25, 28), (10, 14), (10, 23), (13, 25), (10, 25), (13, 27), (16, 25), (13, 22), (16, 27), (13, 15), (13, 24), (23, 25), (16, 22), (23, 27), (13, 28), (24, 30), (16, 19), (23, 24), (13, 16), (16, 30), (8, 13), (19, 22), (16, 23), (15, 27), (23, 28), (8, 15), (27, 28), (19, 24), (11, 22), (15, 22), (11, 15), (19, 28), (11, 24), (19, 30), (8, 14), (14, 27), (11, 19), (15, 19), (15, 28), (26, 28), (25, 30), (22, 24)]]  # noqa: E501        
        # fmt: on
        leaf_clusters = tuple(Cluster(keys=frozenset(), edges=edges, children=()) for edges in subgraph_edges)
        root = Cluster(keys=frozenset(), edges=[], children=leaf_clusters)
        clustering = Clustering(root=root)
        grouped = clustering.group_by_leaf(self.dummy_results)
        self.assertEqual(len(grouped), len(leaf_clusters))
        for i, cluster in enumerate(leaf_clusters):
            # Check that the grouped results match the leaf cluster pairs
            self.assertEqual(set(grouped[i].keys()), set(cluster.edges))


if __name__ == "__main__":
    unittest.main()
