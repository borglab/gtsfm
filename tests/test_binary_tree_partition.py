"""Unit tests for the BinaryTreePartition graph partitioner.

This module tests the functionality of the BinaryTreePartition class, ensuring that:
- It correctly splits image pairs into leaf partitions.
- The resulting partitions contain valid and non-duplicated edges.
- The binary tree structure is valid.
- Edge cases like empty input are handled gracefully.

Author: Shicong Ma
"""

import unittest
from typing import List, Tuple

import gtsam

from gtsfm.graph_partitioner.binary_tree_partition import BinaryTreeNode, BinaryTreePartition


class TestBinaryTreePartition(unittest.TestCase):
    """Unit tests for BinaryTreePartition."""

    def setUp(self):
        """Set up a simple 4x4 grid graph for testing."""
        self.rows, self.cols = 4, 4
        self.total_nodes = self.rows * self.cols
        self.image_pairs = self._create_grid_edges(self.rows, self.cols)

    def _create_grid_edges(self, rows: int, cols: int) -> List[Tuple[int, int]]:
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

    def test_partition_leaf_count(self):
        """Test that partitioner creates the correct number of leaf partitions."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs(self.image_pairs)
        expected_num_leaves = 2**partitioner.max_depth
        self.assertEqual(len(partitions), expected_num_leaves)

    def test_no_duplicate_undirected_edges(self):
        """Test that undirected edges are not duplicated (e.g., both (u,v) and (v,u))."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs(self.image_pairs)
        undirected_edges = set()
        for partition in partitions:
            for u, v in partition:
                edge = (min(u, v), max(u, v))
                undirected_edges.add(edge)
        self.assertEqual(len(undirected_edges), len(set(undirected_edges)))

    def test_edge_validity(self):
        """Test that all edges are valid integer indices within the image grid."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs(self.image_pairs)
        for partition in partitions:
            for u, v in partition:
                self.assertIsInstance(u, int)
                self.assertIsInstance(v, int)
                self.assertGreaterEqual(u, 0)
                self.assertLess(u, self.total_nodes)
                self.assertGreaterEqual(v, 0)
                self.assertLess(v, self.total_nodes)

    def test_non_empty_partitions(self):
        """Test that at least one partition contains edges."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs(self.image_pairs)
        non_empty_count = sum(1 for p in partitions if len(p) > 0)
        self.assertGreater(non_empty_count, 0)

    def test_empty_input(self):
        """Test that empty image pair input returns an empty partition list."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs([])
        self.assertEqual(partitions, [])

    def test_known_input_partition(self):
        """Test partitioning of a simple known image pair set."""
        image_pairs = [(0, 1), (1, 2), (2, 3)]
        partitioner = BinaryTreePartition(max_depth=1)
        partitions = partitioner.partition_image_pairs(image_pairs)
        inter = partitioner.get_inter_partition_edges()

        # Check number of partitions
        self.assertEqual(len(partitions), 2)

        # Gather all intra-partition and inter-partition edges
        flattened = set((min(u, v), max(u, v)) for p in partitions for u, v in p)

        for inter_edges in inter.values():
            for u, v in inter_edges:
                flattened.add((min(u, v), max(u, v)))

        # All input edges should appear
        for u, v in image_pairs:
            self.assertIn((min(u, v), max(u, v)), flattened)

    def test_binary_tree_structure(self):
        """Test binary tree structure has correct number of leaves and leaf depths."""
        partitioner = BinaryTreePartition(max_depth=3)

        # Use synthetic ordering for test
        ordering = type(
            "FakeOrdering",
            (),
            {"size": lambda self: 8, "at": lambda self, idx: ord("a") + idx},  # returns int ASCII code
        )()

        root = partitioner._build_binary_partition(ordering)

        leaf_nodes = []

        def dfs(node):
            if node.is_leaf():
                leaf_nodes.append(node)
            if node.left:
                dfs(node.left)
            if node.right:
                dfs(node.right)

        dfs(root)
        self.assertEqual(len(leaf_nodes), 2**partitioner.max_depth)
        self.assertTrue(all(n.depth == partitioner.max_depth for n in leaf_nodes))

    def test_build_graphs(self):
        """Test that _build_graphs constructs the correct symbolic and networkx graphs."""
        partitioner = BinaryTreePartition(max_depth=1)
        test_pairs = [(0, 1), (1, 2)]
        sfg, keys, nxg = partitioner._build_graphs(test_pairs)

        # Check symbolic graph has correct number of factors
        self.assertEqual(sfg.size(), len(test_pairs))

        # Check nx graph has correct nodes and edges
        self.assertEqual(
            set(nxg.edges()),
            {(gtsam.symbol("x", 0), gtsam.symbol("x", 1)), (gtsam.symbol("x", 1), gtsam.symbol("x", 2))},
        )
        self.assertEqual(len(nxg.nodes), 3)
        self.assertEqual(len(keys), 3)

    def test_build_binary_partition(self):
        """Test that binary tree is built correctly with specified depth and balanced splitting."""
        partitioner = BinaryTreePartition(max_depth=2)

        # Fake ordering: integers 0 through 7
        ordering = type("FakeOrdering", (), {"size": lambda self: 8, "at": lambda self, idx: idx})()

        root = partitioner._build_binary_partition(ordering)

        # Collect leaf nodes and check depth
        leaf_nodes = []

        def dfs(node):
            if node.is_leaf():
                leaf_nodes.append(node)
            if node.left:
                dfs(node.left)
            if node.right:
                dfs(node.right)

        dfs(root)
        self.assertEqual(len(leaf_nodes), 2**partitioner.max_depth)
        self.assertTrue(all(n.depth == partitioner.max_depth for n in leaf_nodes))
        self.assertEqual(sum(len(n.keys) for n in leaf_nodes), 8)

    def test_compute_leaf_partition_details(self):
        """Test that leaf partitions correctly report intra- and inter-partition edges."""
        partitioner = BinaryTreePartition(max_depth=1)
        image_pairs = [(0, 1), (1, 2), (2, 3)]  # Line graph

        # Build graph and partition tree
        _, _, nxg = partitioner._build_graphs(image_pairs)

        # Manually create a binary tree with leaves split as [0,1] and [2,3]
        left = BinaryTreeNode([gtsam.symbol("x", 0), gtsam.symbol("x", 1)], depth=1)
        right = BinaryTreeNode([gtsam.symbol("x", 2), gtsam.symbol("x", 3)], depth=1)
        root = BinaryTreeNode([], depth=0)
        root.left = left
        root.right = right

        leaf_details, inter_edges_map = partitioner._compute_leaf_partition_details(root, nxg)

        self.assertEqual(len(leaf_details), 2)

        flattened_edges = set()
        for d in leaf_details:
            for u, v in d["intra_partition_edges"]:
                flattened_edges.add((min(u, v), max(u, v)))
        for inter_edges in inter_edges_map.values():
            for u, v in inter_edges:
                flattened_edges.add((min(u, v), max(u, v)))

        for u, v in image_pairs:
            self.assertIn((min(u, v), max(u, v)), flattened_edges)


if __name__ == "__main__":
    unittest.main()
