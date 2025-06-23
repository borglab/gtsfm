import unittest
from typing import List, Tuple

from gtsfm.graph_partitioner.binary_tree_partition import (BinaryTreeNode,
                                                           BinaryTreePartition)


class TestBinaryTreePartition(unittest.TestCase):
    def setUp(self):
        """Set up a simple 4x4 grid graph for testing."""
        self.rows, self.cols = 4, 4
        self.total_nodes = self.rows * self.cols
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
        expected_num_leaves = 2**partitioner.max_depth
        self.assertEqual(len(partitions), expected_num_leaves)

    def test_no_duplicate_undirected_edges(self):
        """Test that undirected edges are not duplicated (u,v) vs (v,u)."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs(self.image_pairs)
        undirected_edges = set()
        for partition in partitions:
            for u, v in partition:
                edge = (min(u, v), max(u, v))
                undirected_edges.add(edge)
        self.assertEqual(len(undirected_edges), len(set(undirected_edges)))

    def test_edge_validity(self):
        """Test that all edges are valid integer indices."""
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
        """Test that empty image pair input returns empty partition list."""
        partitioner = BinaryTreePartition(max_depth=2)
        partitions = partitioner.partition_image_pairs([])
        self.assertEqual(partitions, [])

    def test_known_input_partition(self):
        """Test partitioning of a simple known image pair set."""
        image_pairs = [(0, 1), (1, 2), (2, 3)]
        partitioner = BinaryTreePartition(max_depth=1)
        partitions = partitioner.partition_image_pairs(image_pairs)
        self.assertEqual(len(partitions), 2)

        # All original edges should appear in at least one partition (either internal or shared)
        flattened = set()
        for p in partitions:
            for u, v in p:
                flattened.add((min(u, v), max(u, v)))
        for u, v in image_pairs:
            self.assertIn((min(u, v), max(u, v)), flattened)

    def test_binary_tree_structure(self):
        """Test binary tree structure has correct number of leaves and depth."""
        partitioner = BinaryTreePartition(max_depth=3)
        # Use synthetic ordering for test
        ordering = type(
            "FakeOrdering", (), {"size": lambda self: 8, "at": lambda self, idx: ord("a") + idx}  # returns int ASCII
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


if __name__ == "__main__":
    unittest.main()
