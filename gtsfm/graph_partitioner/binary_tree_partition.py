"""Implementation of a binary tree graph partitioner.

This partitioner recursively partitions image pair graphs into a binary tree
structure up to a specified depth, using METIS-based ordering. Leaf nodes
represent exclusive image keys and associated edge groupings.

Authors: Shicong Ma
"""

from math import ceil, log2
from typing import Dict, List, Optional, Tuple

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

    def __init__(self, max_depth: Optional[int] = None, num_cameras_per_cluster: Optional[int] = None):
        """
        Initialize the BinaryTreePartition object.

        Args:
            max_depth: Maximum depth of the binary tree; results in 2^depth partitions.
            num_cameras_per_cluster: Desired number of cameras per cluster; used to compute max_depth.
        """
        super().__init__(process_name="BinaryTreePartition")

        if max_depth is not None:
            self.max_depth = max_depth
        elif num_cameras_per_cluster is not None:
            self._num_cameras_per_cluster = num_cameras_per_cluster
            self.max_depth = None  # to be inferred later
        else:
            raise ValueError("Either max_depth or num_cameras_per_cluster must be provided")

        self.shared_edge_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def partition_image_pairs(self, image_pairs: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Partition image pairs into subgroups using a binary tree.

        Args:
            image_pairs: List of image index pairs (i, j), where i < j.

        Returns:
            A list of image pair subsets (internal only), one for each leaf.
        """
        if not image_pairs:
            logger.warning("No image pairs provided for partitioning.")
            return []

        all_nodes = set(i for ij in image_pairs for i in ij)
        num_cameras = len(all_nodes)

        if self.max_depth is None:
            self.max_depth = ceil(log2(num_cameras / self._num_cameras_per_cluster))

        symbol_graph, _, nx_graph = self._build_graphs(image_pairs)
        ordering = gtsam.Ordering.MetisSymbolicFactorGraph(symbol_graph)
        binary_tree_root_node = self._build_binary_partition(ordering)

        num_leaves = 2**self.max_depth
        image_pairs_per_partition = [[] for _ in range(num_leaves)]

        partition_details, leaf_node_map = self._compute_leaf_partition_details(binary_tree_root_node, nx_graph)

        logger.info(f"BinaryTreePartition: partitioned into {len(partition_details)} leaf nodes.")

        for i in range(num_leaves):
            edges_exclusive = partition_details[i].get("edges_within_exclusive", [])
            image_pairs_per_partition[i] = edges_exclusive

        for i, part in enumerate(partition_details):
            exclusive_keys = part.get("exclusive_keys", [])
            edges_within = part.get("edges_within_exclusive", [])

            logger.info(
                f"Partition {i}:"
                f"  Exclusive Image Keys ({len(exclusive_keys)}): {sorted(exclusive_keys)}\n"
                f"  Internal Edges ({len(edges_within)}): {edges_within}\n"
            )

        self.shared_edge_map = {(i, j): shared_edges for (i, j), shared_edges in leaf_node_map.items() if shared_edges}

        return image_pairs_per_partition

    def get_shared_edges(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Getter for shared edges between leaf partitions."""
        return self.shared_edge_map

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
    ) -> Tuple[List[Dict], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
        """Recursively traverse the binary tree and return partition details per leaf.

        Args:
            node: Current binary tree node being processed.
            nx_graph: NetworkX graph built from image pairs.

        Returns:
            A tuple:
                - List of dictionaries containing exclusive keys and edges per leaf node.
                - Mapping of (leaf_idx_a, leaf_idx_b) to shared edges.
        """
        leaf_details = []
        shared_edge_map = dict()
        leaf_idx_counter = [0]  # mutable counter to track leaf index
        node_to_idx = dict()

        def dfs(n: BinaryTreeNode) -> List[Dict]:
            if n.is_leaf():
                idx = leaf_idx_counter[0]
                leaf_idx_counter[0] += 1
                node_to_idx[n] = idx
                exclusive_keys = set(n.keys)
                return [
                    {
                        "exclusive_keys": [gtsam.Symbol(u).index() for u in exclusive_keys],
                        "edges_within_exclusive": [
                            (gtsam.Symbol(u).index(), gtsam.Symbol(v).index())
                            for u, v in nx_graph.edges()
                            if u in exclusive_keys and v in exclusive_keys
                        ],
                    }
                ]

            left_part = dfs(n.left)
            right_part = dfs(n.right)

            if n.left.is_leaf() and n.right.is_leaf():
                left_keys = set(n.left.keys)
                right_keys = set(n.right.keys)
                shared_edges = [
                    (gtsam.Symbol(u).index(), gtsam.Symbol(v).index())
                    for u, v in nx_graph.edges()
                    if (u in left_keys and v in right_keys) or (u in right_keys and v in left_keys)
                ]
                i = node_to_idx[n.left]
                j = node_to_idx[n.right]
                shared_edge_map[(i, j)] = shared_edges

            return left_part + right_part

        leaf_details = dfs(node)
        return leaf_details, shared_edge_map
