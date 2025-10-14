"""Implementation of a binary tree graph partitioner.

This partitioner recursively partitions a visibility graph into a binary tree
structure up to a specified depth, using METIS-based ordering. Leaf nodes
represent parts with no vertex overlap (i.e., a partition!).

Authors: Shicong Ma and Frank Dellaert
"""

from math import ceil, log2
from typing import List, Optional

from gtsam import Ordering, SymbolicFactorGraph  # type: ignore

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.products.partition import Partition, Subgraph
from gtsfm.products.visibility_graph import VisibilityGraph

logger = logger_utils.get_logger()


class BinaryTreeNode:
    """Node class for a binary tree representing partitioned visibility graphs."""

    def __init__(
        self,
        keys: List[int],
        depth: int,
        left: Optional["BinaryTreeNode"] = None,
        right: Optional["BinaryTreeNode"] = None,
    ):
        """
        Initialize a BinaryTreeNode.

        Args:
            keys: Image indices at this node (only populated at leaf level).
            depth: Depth level in the binary tree.
        """
        self.keys = keys  # Only at leaves
        self.left = left
        self.right = right
        self.depth = depth

    def is_leaf(self) -> bool:
        """Check whether this node is a leaf node."""
        return self.left is None and self.right is None


class BinaryTreePartition(GraphPartitionerBase):
    """Graph partitioner that uses a binary tree to recursively divide a visibility graph."""

    def __init__(self, max_depth: Optional[int] = None, num_cameras_per_cluster: Optional[int] = None):
        """
        Initialize the BinaryTreePartition object.

        Args:
            max_depth: Maximum depth of the binary tree; results in 2^depth partitions.
            num_cameras_per_cluster: Desired number of cameras per cluster; used to compute max_depth.
        """
        super().__init__(process_name="BinaryTreePartition")

        self.max_depth = max_depth
        if max_depth is None:
            if num_cameras_per_cluster is None:
                raise ValueError("Either max_depth or num_cameras_per_cluster must be provided")
            # self.max_depth to be inferred later
            self._num_cameras_per_cluster = num_cameras_per_cluster

    def run(self, graph: VisibilityGraph) -> Partition:
        """Partition visibility graph into subgraphs using a binary tree.

        Args:
            graph: a visibility graph.

        Returns:
            Partition: dataclass with subgraphs and inter-partition edges map.
        """
        if not graph:
            logger.warning("No visibility graph provided for partitioning.")
            return Partition([], {})

        # Check that all pairs are (i, j) with i < j
        for i, j in graph:
            if not i < j:
                raise ValueError(f"VisibilityGraph contains invalid pair ({i}, {j}): i must be less than j.")

        all_nodes = set(i for ij in graph for i in ij)
        num_cameras = len(all_nodes)

        max_depth = self.max_depth
        if max_depth is None:
            max_depth = ceil(log2(num_cameras / self._num_cameras_per_cluster))

        sfg = self._build_symbolic_factor_graph(graph)
        ordering = Ordering.MetisSymbolicFactorGraph(sfg)
        binary_tree_root_node = self._build_binary_partition(ordering, max_depth)

        return self._compute_partition(binary_tree_root_node, graph)

    def _build_symbolic_factor_graph(self, graph: VisibilityGraph) -> SymbolicFactorGraph:
        """Construct GTSAM graph from visibility graph.

        Args:
            graph: List of image index pairs.

        Returns:
            A SymbolicFactorGraph.
        """
        sfg = SymbolicFactorGraph()
        for i, j in graph:
            sfg.push_factor(i, j)
        return sfg

    def _build_binary_partition(self, ordering: Ordering, max_depth: int) -> BinaryTreeNode:
        """Build a binary tree of image keys based on a given ordering.

        Args:
            ordering: GTSAM Ordering object created via METIS.

        Returns:
            Root node of the binary tree.
        """
        ordered_keys = [ordering.at(i) for i in range(ordering.size())]

        def split(keys: List[int], depth: int) -> BinaryTreeNode:
            if depth == max_depth:
                return BinaryTreeNode(keys, depth)

            # NOTE(Frank): this is totally wrong: we should use Metis post-ordering
            # to split the keys into two sets with minimal edge cuts.
            mid = len(keys) // 2
            left_node = split(keys[:mid], depth + 1)
            right_node = split(keys[mid:], depth + 1)
            return BinaryTreeNode([], depth, left_node, right_node)

        return split(ordered_keys, 0)

    def _compute_partition(self, node: BinaryTreeNode, graph: VisibilityGraph) -> Partition:
        """Recursively traverse the binary tree and return partition details per leaf.

        Args:
            node: Current binary tree node being processed.
            graph: Visibility graph.
        Returns:
            A tuple:
                - List of Partition objects per leaf node.
                - Mapping of (leaf_idx_a, leaf_idx_b) to inter-partition edges.
        """
        edge_cuts = dict()
        leaf_idx_counter = [0]
        node_to_idx = dict()

        def dfs(node: BinaryTreeNode) -> List[Subgraph]:
            if node.is_leaf():
                idx = leaf_idx_counter[0]
                leaf_idx_counter[0] += 1
                node_to_idx[node] = idx
                return [
                    Subgraph(
                        keys=set(node.keys),
                        edges=[(i, j) for i, j in graph if i in node.keys and j in node.keys],
                    )
                ]

            assert node.left is not None and node.right is not None
            left_part = dfs(node.left)
            right_part = dfs(node.right)

            # Identify inter-partition edges between two sibling leaf nodes,
            # and map them using their assigned leaf indices.
            if node.left.is_leaf() and node.right.is_leaf():
                left_keys_set = set(node.left.keys)
                right_keys_set = set(node.right.keys)

                shared_edges = []
                for i, j in graph:
                    if (i in left_keys_set and j in right_keys_set) or (i in right_keys_set and j in left_keys_set):
                        shared_edges.append((i, j))

                left_index = node_to_idx[node.left]
                right_index = node_to_idx[node.right]
                edge_cuts[(left_index, right_index)] = shared_edges

            return left_part + right_part

        subgraphs = dfs(node)
        return Partition(subgraphs, edge_cuts)
