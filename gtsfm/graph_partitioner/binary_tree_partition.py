"""Implementation of a binary tree graph partitioner.

This partitioner recursively partitions a visibility graph into a binary tree
structure up to a specified depth, using METIS-based ordering. Leaf nodes
represent subgraphs with no vertex overlap (i.e., a partition!).

Authors: Shicong Ma and Frank Dellaert
"""

from dataclasses import dataclass
from math import ceil, log2
from typing import Dict, List, Optional, Set, Tuple

from gtsam import Ordering, SymbolicFactorGraph  # type: ignore

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
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


@dataclass(frozen=True)
class LeafPartitionDetails:
    exclusive_keys: Set[int]
    intra_partition_edges: VisibilityGraph


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

        self.inter_partition_edges_map: Dict[Tuple[int, int], VisibilityGraph] = {}

    def run(self, graph: VisibilityGraph) -> List[VisibilityGraph]:
        """Partition visibility graph into subgraphs using a binary tree.

        Args:
            graph: a visibility graph.

        Returns:
            A list of visibility graphs (subgraphs), one for each leaf.
        """
        if not graph:
            logger.warning("No visibility graph provided for partitioning.")
            return []

        # Check that all pairs are (i, j) with i < j
        for i, j in graph:
            if not i < j:
                raise ValueError(f"VisibilityGraph contains invalid pair ({i}, {j}): i must be less than j.")

        all_nodes = set(i for ij in graph for i in ij)
        num_cameras = len(all_nodes)

        if self.max_depth is None:
            self.max_depth = ceil(log2(num_cameras / self._num_cameras_per_cluster))

        sfg = self._build_symbolic_factor_graph(graph)
        ordering = Ordering.MetisSymbolicFactorGraph(sfg)
        binary_tree_root_node = self._build_binary_partition(ordering)

        num_leaves = 2**self.max_depth
        image_pairs_per_partition: List[VisibilityGraph] = [[] for _ in range(num_leaves)]

        partition_details, inter_partition_edges = self._compute_leaf_partition_details(binary_tree_root_node, graph)

        logger.info("%d leaf nodes.", len(partition_details))

        for i in range(num_leaves):
            intra_partition_edges = partition_details[i].intra_partition_edges
            image_pairs_per_partition[i] = intra_partition_edges

        for i, part in enumerate(partition_details):
            exclusive_keys = part.exclusive_keys
            intra_edges = part.intra_partition_edges

            logger.info("Partition %d: keys (%d): %s", i + 1, len(exclusive_keys), exclusive_keys)
            logger.info("Partition %d: num intra-partition edges: %d", i + 1, len(intra_edges))
            logger.debug("Partition %d: intra-partition edges: %s", i + 1, intra_edges)

        self.inter_partition_edges_map = {(i, j): edges for (i, j), edges in inter_partition_edges.items() if edges}

        return image_pairs_per_partition

    def get_inter_partition_edges(self) -> Dict[Tuple[int, int], VisibilityGraph]:
        """Getter for inter-partition edges between leaf partitions.

        Returns:
            A dictionary mapping (leaf_idx_a, leaf_idx_b) to a list of edges
            connecting nodes in the two leaf partitions. Only sibling pairs are included.
        """
        return self.inter_partition_edges_map

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

    def _build_binary_partition(self, ordering: Ordering) -> BinaryTreeNode:
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

            # NOTE(Frank): this is totally wrong: we should use Metis post-ordering
            # to split the keys into two sets with minimal edge cuts.
            mid = len(keys) // 2
            left_node = split(keys[:mid], depth + 1)
            right_node = split(keys[mid:], depth + 1)
            return BinaryTreeNode([], depth, left_node, right_node)

        return split(ordered_keys, 0)

    def _compute_leaf_partition_details(
        self, node: BinaryTreeNode, graph: VisibilityGraph
    ) -> Tuple[List[LeafPartitionDetails], Dict[Tuple[int, int], VisibilityGraph]]:
        """Recursively traverse the binary tree and return partition details per leaf.

        Args:
            node: Current binary tree node being processed.
            graph: Visibility graph.
        Returns:
            A tuple:
                - List of LeafPartitionDetails objects per leaf node.
                - Mapping of (leaf_idx_a, leaf_idx_b) to inter-partition edges.
        """
        leaf_details = []
        inter_partition_edges = dict()
        leaf_idx_counter = [0]
        node_to_idx = dict()

        def dfs(node: BinaryTreeNode) -> List[LeafPartitionDetails]:
            if node.is_leaf():
                idx = leaf_idx_counter[0]
                leaf_idx_counter[0] += 1
                node_to_idx[node] = idx
                exclusive_keys = set(node.keys)
                return [
                    LeafPartitionDetails(
                        exclusive_keys=exclusive_keys,
                        intra_partition_edges=[(i, j) for i, j in graph if i in exclusive_keys and j in exclusive_keys],
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
                inter_partition_edges[(left_index, right_index)] = shared_edges

            return left_part + right_part

        leaf_details = dfs(node)
        return leaf_details, inter_partition_edges
