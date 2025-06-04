from typing import List, Tuple

import metis
import networkx as nx

from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.utils.logger import get_logger

logger = get_logger()


class BinaryPartitionTreeNode:
    """Represents a node in the binary partition tree."""

    def __init__(self, graph_edges: List[Tuple[int, int]], nodes: List[int]):
        self.graph_edges = graph_edges
        self.nodes = nodes
        self.left = None
        self.right = None


class BinaryPartitionTree:
    """Binary tree structure for recursive graph partitioning."""

    def __init__(self, graph: nx.Graph, max_levels=3):
        self.max_levels = max_levels
        self.root = self._partition_graph(graph, level=0)

    def _partition_graph(self, graph: nx.Graph, level: int) -> BinaryPartitionTreeNode:
        if level == self.max_levels or len(graph.nodes()) <= 1:
            return BinaryPartitionTreeNode(list(graph.edges()), list(graph.nodes()))

        edgecuts, partitions = metis.part_graph(graph, nparts=2, recursive=True)
        partition_0 = [node for i, node in enumerate(graph.nodes()) if partitions[i] == 0]
        partition_1 = [node for i, node in enumerate(graph.nodes()) if partitions[i] == 1]

        subgraph_0 = graph.subgraph(partition_0).copy()
        subgraph_1 = graph.subgraph(partition_1).copy()

        node = BinaryPartitionTreeNode(list(graph.edges()), list(graph.nodes()))
        node.left = self._partition_graph(subgraph_0, level + 1)
        node.right = self._partition_graph(subgraph_1, level + 1)

        return node

    def get_leaf_partitions(self) -> List[List[Tuple[int, int]]]:
        leaf_partitions = []

        def collect_leaves(node: BinaryPartitionTreeNode):
            if node is None:
                return
            if node.left is None and node.right is None:
                leaf_partitions.append(node.graph_edges)
            else:
                collect_leaves(node.left)
                collect_leaves(node.right)

        collect_leaves(self.root)
        return leaf_partitions


class EightPartitions(GraphPartitionerBase):
    """Graph partitioner that returns eight edge partitions using recursive METIS."""

    def __init__(self):
        super().__init__(process_name="EightPartitions")

    def partition_image_pairs(self, image_pairs: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """Partition image pairs into 8 subgraphs using a 3-level binary tree.

        Args:
            image_pairs: List of image pairs (i,j) where i < j.

        Returns:
            A list of 8 sublists, each containing a subset of image pairs.
        """
        G = nx.Graph()
        G.add_edges_from(image_pairs)

        tree = BinaryPartitionTree(G, max_levels=3)
        leaf_edge_lists = tree.get_leaf_partitions()

        logger.info(f"EightPartition: partitioned into {len(leaf_edge_lists)} subgraphs")
        return leaf_edge_lists
