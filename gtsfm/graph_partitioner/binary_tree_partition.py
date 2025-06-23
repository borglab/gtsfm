from typing import Dict, List, Tuple

import gtsam
import networkx as nx
from gtsam import SymbolicFactorGraph

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase

logger = logger_utils.get_logger()


class BinaryTreeNode:
    def __init__(self, keys: List[int], depth: int):
        self.keys = keys  # Only at leaves
        self.left = None
        self.right = None
        self.depth = depth

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class BinaryTreePartition(GraphPartitionerBase):
    def __init__(self, max_depth: int = 2):
        super().__init__(process_name="BinaryTreePartition")
        self.max_depth = max_depth

    def partition_image_pairs(self, image_pairs: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        if not image_pairs:
            logger.warning("No image pairs provided for partitioning.")
            return []

        symbol_graph, _, nx_graph = self._build_graphs(image_pairs)
        ordering = gtsam.Ordering.MetisSymbolicFactorGraph(symbol_graph)
        self.root = self._build_binary_partition(ordering)

        num_leaves = 2**self.max_depth
        partition_details = [{} for _ in range(num_leaves)]
        image_pair_partitions = [[] for _ in range(num_leaves)]

        self._collect_leaf_partitions(self.root, nx_graph, partition_details, leaf_idx=[0])

        logger.info(f"BinaryTreePartition: partitioned into {len(partition_details)} leaf nodes.")

        for i in range(num_leaves):
            edges_explicit = partition_details[i].get("edges_within_explicit", [])
            edges_shared = partition_details[i].get("edges_with_shared", [])
            image_pair_partitions[i] = edges_explicit + edges_shared

        for i, part in enumerate(partition_details):
            explicit_keys = part.get("explicit_keys", [])
            edges_within = part.get("edges_within_explicit", [])
            edges_shared = part.get("edges_with_shared", [])

            logger.info(
                f"Partition {i}:\n"
                f"  Explicit Keys ({len(explicit_keys)}): {sorted(explicit_keys)}\n"
                f"  Internal Edges ({len(edges_within)}): {edges_within}\n"
                f"  Shared Edges   ({len(edges_shared)})"
            )

        return image_pair_partitions

    def _build_graphs(self, image_pairs: List[Tuple[int, int]]) -> Tuple[SymbolicFactorGraph, List[int], nx.Graph]:
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

    def _collect_leaf_partitions(
        self,
        node: BinaryTreeNode,
        nx_graph: nx.Graph,
        partitions: List[Dict],
        leaf_idx: List[int],
    ):
        if node.is_leaf():
            idx = leaf_idx[0]
            leaf_idx[0] += 1

            explicit_keys = set(node.keys)
            edges_within_explicit = []
            edges_with_shared = []

            for u, v in nx_graph.edges():
                if u in explicit_keys and v in explicit_keys:
                    edges_within_explicit.append((gtsam.Symbol(u).index(), gtsam.Symbol(v).index()))
                elif u in explicit_keys or v in explicit_keys:
                    edges_with_shared.append((gtsam.Symbol(u).index(), gtsam.Symbol(v).index()))

            partitions[idx] = {
                "explicit_keys": [gtsam.Symbol(u).index() for u in explicit_keys],
                "explicit_count": len(explicit_keys),
                "edges_within_explicit": edges_within_explicit,
                "edges_with_shared": edges_with_shared,
            }
            return

        # Recursively collect for children, compute shared variables
        self._collect_leaf_partitions(node.left, nx_graph, partitions, leaf_idx)
        self._collect_leaf_partitions(node.right, nx_graph, partitions, leaf_idx)
