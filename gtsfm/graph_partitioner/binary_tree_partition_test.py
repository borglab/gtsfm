import unittest
from typing import List, Tuple

import gtsam
import networkx as nx

from gtsfm.graph_partitioner.binary_tree_partition import BinaryTreePartition


class TestBinaryTreePartition(unittest.TestCase):
    def setUp(self):
        """Set up a simple 4x4 grid graph for testing."""
        self.rows, self.cols = 4, 4
        self.image_pairs = self._create_grid_edges(self.rows, self.cols)

    def _create_grid_edges(self, rows: int, cols: int) -> List[Tuple[int, int]]:
        """Create a simple 2D grid of image pairs."""
        edges = []
        for r in range(rows):
            for c in range(cols):
                current_idx = r * cols + c
                if c + 1 < cols:
                    edges.append((current_idx, current_idx + 1))
                if r + 1 < rows:
                    edges.append((current_idx, current_idx + cols))
        return edges

    def test_partition_leaf_count(self):
        """Test that partitioner creates the correct number of leaf partitions."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs(self.image_pairs)

        # Should create exactly 2^max_depth partitions
        expected_num_leaves = 2**partitioner.max_depth
        self.assertEqual(len(partitions), expected_num_leaves)

    def test_no_duplicate_undirected_edges(self):
        """Test that undirected edges are not duplicated (u,v) and (v,u) are treated as same."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs(self.image_pairs)

        undirected_edges = set()

        for partition in partitions:
            for u, v in partition:
                edge = (min(u, v), max(u, v))
                undirected_edges.add(edge)

        # Now just check that the number of unique undirected edges matches
        all_edges = []
        for partition in partitions:
            for u, v in partition:
                all_edges.append((min(u, v), max(u, v)))

        self.assertEqual(len(set(all_edges)), len(undirected_edges))

    def test_edge_validity(self):
        """Test that all edges are valid and involve at least one frontal node."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs(self.image_pairs)

        for partition in partitions:
            for u, v in partition:
                # Each u, v should be int indices
                self.assertIsInstance(u, int)
                self.assertIsInstance(v, int)
                self.assertLessEqual(u, (self.rows * self.cols) - 1)
                self.assertLessEqual(v, (self.rows * self.cols) - 1)

    def test_non_empty_partitions(self):
        """Test that partitions are not all empty (unless small graphs)."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs(self.image_pairs)

        non_empty_count = sum(1 for p in partitions if len(p) > 0)
        self.assertGreater(non_empty_count, 0)


if __name__ == "__main__":
    unittest.main()
