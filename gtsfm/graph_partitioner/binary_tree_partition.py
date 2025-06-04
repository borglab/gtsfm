from typing import List, Tuple

import graphviz
import gtsam
import networkx as nx
from gtsam import SymbolicFactorGraph

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase

logger = logger_utils.get_logger()


class BinaryTreeNode:
    def __init__(self, frontals: List[int], separators: List[int], depth: int):
        self.frontals = frontals
        self.separators = separators
        self.left = None
        self.right = None
        self.depth = depth

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class BinaryTreePartition(GraphPartitionerBase):
    """Graph partitioner that partitions image pairs using a binary tree built from METIS ordering."""

    def __init__(self, max_depth: int = 2):
        """Initialize the partitioner."""
        super().__init__(process_name="BinaryTreePartition")
        self.max_depth = max_depth

    def partition_image_pairs(self, image_pairs: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Partition image pairs into leaf nodes of a binary tree."""
        if not image_pairs:
            logger.warning("No image pairs provided for partitioning.")
            return []

        # Build graphs
        symbol_graph, symbols, nx_graph = self._build_graphs(image_pairs)

        # Run METIS ordering
        ordering = gtsam.Ordering.MetisSymbolicFactorGraph(symbol_graph)

        # Build binary tree
        root = self._build_binary_partition(ordering, nx_graph)

        # Collect partitions from leaf nodes
        partitions = [[] for _ in range(pow(2, self.max_depth))]
        self._collect_leaf_partitions(root, nx_graph, partitions, leaf_idx=[0])

        logger.info(f"BinaryTreePartition: partitioned into {len(partitions)} leaf nodes.")
        return partitions

    def _build_graphs(self, image_pairs: List[Tuple[int, int]]) -> Tuple[SymbolicFactorGraph, List[int], nx.Graph]:
        """Create a SymbolicFactorGraph and NetworkX graph from image pairs."""
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

    def _build_binary_partition(self, ordering: gtsam.Ordering, nx_graph: nx.Graph) -> BinaryTreeNode:
        """Recursively build a binary tree from METIS ordering."""
        ordered_keys = [ordering.at(i) for i in range(ordering.size())]

        def split(keys: List[int], depth: int, parent_separator: List[int]) -> BinaryTreeNode:
            node = BinaryTreeNode(frontals=keys, separators=parent_separator, depth=depth)

            if depth == self.max_depth or len(keys) <= 3:
                return node

            mid = len(keys) // 2
            left_keys = keys[:mid]
            right_keys = keys[mid:]

            if len(left_keys) < 3 or len(right_keys) < 3:
                return node

            separators = self._find_separators(nx_graph, left_keys, right_keys)

            if len(separators) == 0:
                return node

            node.left = split(left_keys, depth + 1, separators)
            node.right = split(right_keys, depth + 1, separators)

            return node

        return split(ordered_keys, depth=0, parent_separator=[])

    def _find_separators(self, nx_graph: nx.Graph, group_a: List[int], group_b: List[int]) -> List[int]:
        """Find separator variables connecting group_a and group_b."""
        separators = set()
        group_a_set = set(group_a)
        group_b_set = set(group_b)

        for u, v in nx_graph.edges():
            if (u in group_a_set and v in group_b_set) or (v in group_a_set and u in group_b_set):
                separators.add(u)
                separators.add(v)

        return list(separators)

    def _collect_leaf_partitions(
        self,
        node: BinaryTreeNode,
        nx_graph: nx.Graph,
        partitions: List[List[Tuple[int, int]]],
        leaf_idx: List[int],
    ):
        """Collect leaf node partitions based on node's own separator."""
        if node.is_leaf():
            print(f"Collecting for leaf {leaf_idx[0]}")

            frontal_set = set(node.frontals)
            separator_set = set(node.separators)
            full_node_set = frontal_set | separator_set

            print(f"  Frontals: {node.frontals}")
            print(f"  Separators: {node.separators}")
            print(f"  Full nodes: {full_node_set}")

            subgraph_edges = []

            for u, v in nx_graph.edges():
                if u in full_node_set and v in full_node_set:
                    u_in_frontals = u in frontal_set
                    v_in_frontals = v in frontal_set

                    if u_in_frontals or v_in_frontals:
                        u_idx = gtsam.Symbol(u).index()
                        v_idx = gtsam.Symbol(v).index()
                        subgraph_edges.append((min(u_idx, v_idx), max(u_idx, v_idx)))

            partitions[leaf_idx[0]] = subgraph_edges
            leaf_idx[0] += 1
            return

        if node.left:
            self._collect_leaf_partitions(node.left, nx_graph, partitions, leaf_idx)
        if node.right:
            self._collect_leaf_partitions(node.right, nx_graph, partitions, leaf_idx)
