"""Unit tests for the BinaryTreePartitioner graph partitioner.

This module tests the functionality of the BinaryTreePartitioner class, ensuring that:
- It correctly splits image pairs into leaf clusters.
- The resulting clusters contain valid and non-duplicated edges.
- The binary tree structure captures inter-cluster edges.
- Edge cases like empty input are handled gracefully.
"""

import unittest
from typing import Set, Tuple

from gtsfm.graph_partitioner.binary_tree_partitioner import BinaryTreePartitioner
from gtsfm.products.cluster_tree import Cluster
from gtsfm.products.visibility_graph import ImageIndexPairs


class TestBinaryTreePartitioner(unittest.TestCase):
    """Unit tests for BinaryTreePartitioner."""

    def setUp(self):
        """Set up a simple 4x4 grid graph for testing."""
        self.rows, self.cols = 4, 4
        self.total_nodes = self.rows * self.cols
        self.image_pairs = self._create_grid_edges(self.rows, self.cols)

    def _create_grid_edges(self, rows: int, cols: int) -> ImageIndexPairs:
        """Create a simple 2D grid of image pairs.

        Args:
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.

        Returns:
            List of edges between adjacent grid points.
        """
        edges = []
        for r in range(rows):
            for c in range(cols):
                current_idx = r * cols + c
                if c + 1 < cols:
                    edges.append((current_idx, current_idx + 1))
                if r + 1 < rows:
                    edges.append((current_idx, current_idx + cols))
        return edges

    def _collect_all_edges(self, cluster: Cluster) -> Set[Tuple[int, int]]:
        edges = set(cluster.edges)
        for child in cluster.children:
            edges.update(self._collect_all_edges(child))
        return edges

    def test_leaf_count(self):
        """Test that partitioner creates the expected number of leaf clusters."""
        partitioner = BinaryTreePartitioner(max_depth=2)
        cluster_tree = partitioner.run(self.image_pairs)
        self.assertFalse(cluster_tree.is_empty())
        leaves = cluster_tree.leaves()
        assert partitioner.max_depth is not None
        expected_num_leaves = 2**partitioner.max_depth
        self.assertEqual(len(leaves), expected_num_leaves)

    def test_no_duplicate_edges_in_leaves(self):
        """Test that undirected edges are not duplicated across leaves."""
        partitioner = BinaryTreePartitioner(max_depth=2)
        cluster_tree = partitioner.run(self.image_pairs)
        seen_edges = set()
        for cluster in cluster_tree.leaves():
            for u, v in cluster.edges:
                edge = (min(u, v), max(u, v))
                self.assertNotIn(edge, seen_edges)
                seen_edges.add(edge)

    def test_edge_validity(self):
        """Test that all edges are valid integer indices within the image grid."""
        partitioner = BinaryTreePartitioner(max_depth=2)
        cluster_tree = partitioner.run(self.image_pairs)
        self.assertIsNotNone(cluster_tree.root)
        all_edges = self._collect_all_edges(cluster_tree.root)  # type: ignore[arg-type]
        for u, v in all_edges:
            self.assertIsInstance(u, int)
            self.assertIsInstance(v, int)
            self.assertGreaterEqual(u, 0)
            self.assertLess(u, self.total_nodes)
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, self.total_nodes)

    def test_non_empty_leaf(self):
        """Test that at least one leaf cluster contains edges."""
        partitioner = BinaryTreePartitioner(max_depth=2)
        cluster_tree = partitioner.run(self.image_pairs)
        non_empty_count = sum(1 for c in cluster_tree.leaves() if len(c.edges) > 0)
        self.assertGreater(non_empty_count, 0)

    def test_empty_input_returns_empty_clustering(self):
        """Test that empty image pair input returns an empty cluster tree."""
        partitioner = BinaryTreePartitioner(max_depth=2)
        cluster_tree = partitioner.run([])
        self.assertTrue(cluster_tree.is_empty())
        self.assertEqual(cluster_tree.leaves(), ())

    def test_known_input_edges_distributed(self):
        """Test cluster tree of a simple known image pair set."""
        image_pairs = [(0, 1), (1, 2), (2, 3)]
        partitioner = BinaryTreePartitioner(max_depth=1)
        cluster_tree = partitioner.run(image_pairs)

        self.assertFalse(cluster_tree.is_empty())
        assert cluster_tree.root is not None

        # All edges should appear somewhere in the hierarchy.
        all_edges = self._collect_all_edges(cluster_tree.root)
        self.assertSetEqual(all_edges, set(image_pairs))

        # Cross-cluster edge should be stored at the root.
        self.assertIn((1, 2), cluster_tree.root.edges)


if __name__ == "__main__":
    unittest.main()
