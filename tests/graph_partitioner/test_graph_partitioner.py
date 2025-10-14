"""Unit tests for graph partitioning functionality.

Authors: Zongyue Liu
"""

import unittest

import numpy as np

import gtsfm.utils.subgraph_utils as subgraph_utils
from gtsfm.graph_partitioner.binary_tree_partition import BinaryTreePartition
from gtsfm.graph_partitioner.single_partition import SinglePartition


class TestGraphPartitioning(unittest.TestCase):
    """Tests for graph partitioning functionality."""

    def setUp(self) -> None:
        # fmt: off
        # From Skydio-32 dataset:
        messed_up_pairs = [(7, 17), (15, 30), (4, 9), (3, 13), (8, 18), (0, 14), (np.int64(11), np.int64(23)), (7, 10), (np.int64(15), np.int64(23)), (np.int64(18), np.int64(19)), (3, 15), (22, 28), (8, 11), (14, 24), (11, 16), (15, 25), (np.int64(7), np.int64(21)), (3, 8), (14, 17), (np.int64(0), np.int64(9)), (np.int64(15), np.int64(18)), (3, 10), (np.int64(0), np.int64(2)), (14, 19), (np.int64(14), np.int64(28)), (10, 24), (7, 16), (np.int64(29), np.int64(31)), (3, 12), (np.int64(22), np.int64(25)), (21, 27), (10, 17), (2, 13), (3, 14), (14, 23), (np.int64(2), np.int64(6)), (17, 25), (7, 11), (6, 15), (np.int64(3), np.int64(7)), (np.int64(3), np.int64(16)), (np.int64(14), np.int64(16)), (14, 25), (10, 12), (2, 8), (np.int64(10), np.int64(30)), (25, 28), (np.int64(6), np.int64(17)), (3, 9), (14, 18), (np.int64(17), np.int64(20)), (10, 14), (np.int64(10), np.int64(23)), (2, 10), (13, 25), (6, 19), (17, 22), (9, 18), (17, 31), (10, 25), (np.int64(13), np.int64(27)), (np.int64(3), np.int64(4)), (6, 21), (np.int64(9), np.int64(11)), (17, 24), (10, 18), (2, 14), (np.int64(9), np.int64(29)), (13, 20), (6, 14), (np.int64(24), np.int64(29)), (16, 25), (5, 15), (2, 7), (np.int64(9), np.int64(22)), (np.int64(6), np.int64(7)), (13, 22), (np.int64(6), np.int64(16)), (np.int64(16), np.int64(18)), (np.int64(24), np.int64(31)), (16, 27), (5, 8), (np.int64(5), np.int64(17)), (np.int64(17), np.int64(19)), (9, 15), (2, 9), (np.int64(13), np.int64(15)), (np.int64(6), np.int64(9)), (13, 24), (6, 18), (np.int64(16), np.int64(20)), (5, 10), (np.int64(23), np.int64(25)), (5, 19), (9, 17), (17, 30), (1, 15), (13, 17), (6, 11), (16, 22), (12, 18), (5, 12), (np.int64(2), np.int64(4)), (17, 23), (23, 27), (np.int64(13), np.int64(28)), (5, 14), (np.int64(23), np.int64(29)), (9, 12), (1, 10), (27, 29), (13, 21), (np.int64(16), np.int64(17)), (np.int64(24), np.int64(30)), (np.int64(5), np.int64(7)), (12, 22), (np.int64(5), np.int64(16)), (4, 18), (np.int64(12), np.int64(31)), (np.int64(23), np.int64(31)), (np.int64(1), np.int64(3)), (27, 31), (1, 12), (np.int64(16), np.int64(19)), (np.int64(5), np.int64(9)), (np.int64(23), np.int64(24)), (5, 18), (9, 16), (1, 5), (np.int64(8), np.int64(20)), (13, 16), (19, 29), (np.int64(16), np.int64(21)), (16, 30), (12, 17), (5, 11), (8, 13), (1, 7), (19, 22), (1, 16), (16, 23), (15, 27), (np.int64(4), np.int64(6)), (12, 19), (5, 13), (4, 15), (23, 28), (8, 15), (np.int64(27), np.int64(28)), (1, 9), (0, 11), (19, 24), (18, 25), (4, 8), (np.int64(1), np.int64(2)), (0, 4), (8, 17), (np.int64(0), np.int64(13)), (11, 22), (np.int64(11), np.int64(31)), (15, 22), (7, 18), (4, 10), (12, 23), (4, 19), (1, 4), (np.int64(0), np.int64(6)), (0, 15), (11, 15), (19, 28), (np.int64(11), np.int64(24)), (np.int64(7), np.int64(20)), (np.int64(18), np.int64(20)), (12, 16), (18, 29), (4, 12), (8, 12), (1, 6), (0, 8), (np.int64(8), np.int64(21)), (19, 21), (19, 30), (np.int64(15), np.int64(17)), (7, 13), (np.int64(7), np.int64(22)), (18, 22), (4, 14), (18, 31), (np.int64(0), np.int64(1)), (22, 31), (8, 14), (np.int64(14), np.int64(27)), (0, 10), (11, 19), (15, 19), (7, 15), (np.int64(15), np.int64(28)), (np.int64(26), np.int64(28)), (25, 30), (np.int64(4), np.int64(7)), (np.int64(4), np.int64(16)), (np.int64(22), np.int64(24)), (0, 3), (14, 20), (11, 21)]  # noqa: E501
        # fmt: on
        self.dummy_results = subgraph_utils.normalize_keys({key: None for key in messed_up_pairs})
        self.pairs = list(self.dummy_results.keys())

    def test_single_partition(self):
        """Test that SinglePartition correctly returns all pairs as one partition."""
        # Create some dummy image pairs
        image_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]

        # Create a SinglePartition instance
        partitioner = SinglePartition()

        # Get partitioned result
        partitioned_pairs = partitioner.run(image_pairs)

        # Check that we get exactly one partition
        self.assertEqual(len(partitioned_pairs), 1)

        # Check that the partition contains all the original pairs
        self.assertEqual(set(partitioned_pairs[0]), set(image_pairs))

    def test_partition_preserves_key_order(self):
        """Test that all edges in partitions have i < j (valid key order)."""
        partitioner = BinaryTreePartition(max_depth=1)
        partitions = partitioner.run(self.pairs)
        total_edges = 0
        for _, partition in enumerate(partitions):
            total_edges += len(partition)
            for i, j in partition:
                self.assertLess(i, j, f"Edge ({i},{j}) does not satisfy i < j")
        # regression:
        self.assertEqual(total_edges, 112, "Expected 112 intra-edges in total")

    def test_group_results_by_subgraph(self):
        """Test grouping results by subgraph."""
        # fmt: off
        partitions = [[(7, 17), (4, 9), (7, 21), (0, 9), (0, 2), (29, 31), (3, 12), (2, 6), (3, 7), (6, 17), (3, 9), (17, 20), (9, 18), (17, 31), (3, 4), (6, 21), (9, 29), (2, 7), (6, 7), (5, 17), (2, 9), (6, 9), (6, 18), (9, 17), (12, 18), (5, 12), (2, 4), (9, 12), (5, 7), (4, 18), (12, 31), (1, 3), (1, 12), (5, 9), (5, 18), (1, 5), (12, 17), (1, 7), (4, 6), (1, 9), (1, 2), (0, 4), (7, 18), (1, 4), (0, 6), (7, 20), (18, 20), (18, 29), (4, 12), (1, 6), (18, 31), (0, 1), (4, 7), (0, 3)], [(15, 30), (11, 23), (15, 23), (22, 28), (8, 11), (14, 24), (11, 16), (15, 25), (14, 19), (14, 28), (10, 24), (22, 25), (14, 23), (14, 16), (14, 25), (10, 30), (25, 28), (10, 14), (10, 23), (13, 25), (10, 25), (13, 27), (16, 25), (13, 22), (16, 27), (13, 15), (13, 24), (23, 25), (16, 22), (23, 27), (13, 28), (24, 30), (16, 19), (23, 24), (13, 16), (16, 30), (8, 13), (19, 22), (16, 23), (15, 27), (23, 28), (8, 15), (27, 28), (19, 24), (11, 22), (15, 22), (11, 15), (19, 28), (11, 24), (19, 30), (8, 14), (14, 27), (11, 19), (15, 19), (15, 28), (26, 28), (25, 30), (22, 24)]]  # noqa: E501
        # fmt: on
        grouped = subgraph_utils.group_results_by_subgraph(self.dummy_results, partitions)
        self.assertEqual(len(grouped), len(partitions))
        for i in range(len(partitions)):
            # Check that the grouped results match the subgraph pairs
            self.assertEqual(set(grouped[i].keys()), set(partitions[i]))


if __name__ == "__main__":
    unittest.main()
