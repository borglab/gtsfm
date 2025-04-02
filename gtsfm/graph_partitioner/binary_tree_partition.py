"""Implementation of a binary tree graph partitioner.

This partitioner recursively partitions image pair graphs into a binary tree
structure up to a specified depth, using METIS-based ordering. Leaf nodes
represent explicit image keys and associated edge groupings.

Authors: Shicong Ma
"""

from typing import Dict, List, Tuple

import gtsam
import networkx as nx
from gtsam import SymbolicFactorGraph

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase

logger = logger_utils.get_logger()


class BinaryTreeNode:
    """Node class for a binary tree representing partitioned sets of image keys."""

    def __init__(self, keys: List[int], depth: int):
        """
        Initialize a BinaryTreeNode.

        Args:
            keys: Image indices at this node (only populated at leaf level).
            depth: Depth level in the binary tree.
        """
        self.keys = keys  # Only at leaves
        self.left = None
        self.right = None
        self.depth = depth

    def is_leaf(self) -> bool:
        """Check whether this node is a leaf node."""
        return self.left is None and self.right is None


class BinaryTreePartition(GraphPartitionerBase):
    """Graph partitioner that uses a binary tree to recursively divide image pairs."""

    def __init__(self, max_depth: int = 2):
        """
        Initialize the BinaryTreePartition object.

        Args:
            max_depth: Maximum depth of the binary tree; results in 2^depth partitions.
        """
        super().__init__(process_name="BinaryTreePartition")
        self.max_depth = max_depth

    def partition_image_pairs(self, image_pairs: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Partition image pairs into subgroups using a binary tree.

        Args:
            image_pairs: List of image index pairs (i, j), where i < j.

        Returns:
            A list of image pair subsets, one for each leaf in the binary tree.
        """
        if not image_pairs:
            logger.warning("No image pairs provided for partitioning.")
            return []

        symbol_graph, _, nx_graph = self._build_graphs(image_pairs)
        ordering = gtsam.Ordering.MetisSymbolicFactorGraph(symbol_graph)
        binary_tree_root_node = self._build_binary_partition(ordering)

        num_leaves = 2**self.max_depth
        image_pairs_per_partition = [[] for _ in range(num_leaves)]

        partition_details = self._compute_leaf_partition_details(binary_tree_root_node, nx_graph)

        logger.info(f"BinaryTreePartition: partitioned into {len(partition_details)} leaf nodes.")

        for i in range(num_leaves):
            edges_explicit = partition_details[i].get("edges_within_explicit", [])
            edges_shared = partition_details[i].get("edges_with_shared", [])
            image_pairs_per_partition[i] = edges_explicit + edges_shared

        for i, part in enumerate(partition_details):
            explicit_keys = part.get("explicit_keys", [])
            edges_within = part.get("edges_within_explicit", [])
            edges_shared = part.get("edges_with_shared", [])

            logger.info(
                f"Partition {i}:\n"
                f"  Explicit Image Keys that only exist within the current partition "
                f"({len(explicit_keys)}): {sorted(explicit_keys)}\n"
                f"  Internal Edges ({len(edges_within)}): {edges_within}\n"
                f"  Shared Edges   ({len(edges_shared)}): {edges_shared}\n"
            )

        return image_pairs_per_partition

    def _build_graphs(self, image_pairs: List[Tuple[int, int]]) -> Tuple[SymbolicFactorGraph, List[int], nx.Graph]:
        """Construct GTSAM and NetworkX graphs from image pairs.

        Args:
            image_pairs: List of image index pairs.

        Returns:
            A tuple of (SymbolicFactorGraph, list of keys, NetworkX graph).
        """
        sfg = gtsam.SymbolicFactorGraph()
        nxg = nx.Graph()
        keys = set()

        for i, j in image_pairs:
            key_i = gtsam.symbol("x", i)
            key_j = gtsam.symbol("x", j)
            keys.add(key_i)
            keys.add(key_j)

            sfg.push_factor(key_i, key_j)
            nxg.add_edge(key_i, key_j)

        return sfg, list(keys), nxg

    def _build_binary_partition(self, ordering: gtsam.Ordering) -> BinaryTreeNode:
        """Build a binary tree of image keys based on a given ordering.

        Args:
            ordering: GTSAM Ordering object created via METIS.

        Returns:
            Root node of the binary tree.
        """
        ordered_keys = [ordering.at(i) for i in range(ordering.size())]

        def split(keys: List[int], depth: int) -> BinaryTreeNode:
            if depth == self.max_depth:
                return BinaryTreeNode(keys, depth)

            mid = len(keys) // 2
            left_node = split(keys[:mid], depth + 1)
            right_node = split(keys[mid:], depth + 1)
            node = BinaryTreeNode([], depth)
            node.left = left_node
            node.right = right_node
            return node

        return split(ordered_keys, 0)

    def _compute_leaf_partition_details(
        self,
        node: BinaryTreeNode,
        nx_graph: nx.Graph,
    ) -> List[Dict]:
        """Recursively traverse the binary tree and return partition details per leaf.

        Args:
            node: Current binary tree node being processed.
            nx_graph: NetworkX graph built from image pairs.

        Returns:
            A list of dictionaries containing partition details per leaf node.
        """
        if node.is_leaf():
            explicit_keys = set(node.keys)
            return [
                {
                    "explicit_keys": [gtsam.Symbol(u).index() for u in explicit_keys],
                    "explicit_count": len(explicit_keys),
                    "edges_within_explicit": [
                        (gtsam.Symbol(u).index(), gtsam.Symbol(v).index())
                        for u, v in nx_graph.edges()
                        if u in explicit_keys and v in explicit_keys
                    ],
                    "edges_with_shared": [],  # placeholder
                }
            ]

        # Recursively compute for children
        left_partitions = self._compute_leaf_partition_details(node.left, nx_graph)
        right_partitions = self._compute_leaf_partition_details(node.right, nx_graph)

        if node.left.is_leaf() and node.right.is_leaf():
            left_keys = set(node.left.keys)
            right_keys = set(node.right.keys)

            shared_edges = [
                (gtsam.Symbol(u).index(), gtsam.Symbol(v).index())
                for u, v in nx_graph.edges()
                if (u in left_keys and v in right_keys) or (u in right_keys and v in left_keys)
            ]

            # Directly assign shared edges to the only two leaf partitions
            left_partitions[0]["edges_with_shared"] = shared_edges
            right_partitions[0]["edges_with_shared"] = shared_edges

        return left_partitions + right_partitions
