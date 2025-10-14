"""Implementation of a binary tree graph partitioner.

This partitioner recursively clusters a visibility graph into a binary tree
structure up to a specified depth, using METIS-based ordering. Leaf nodes
represent disjoint clusters with no vertex overlap, while internal nodes
capture the inter-cluster edges between their children.

Authors: Shicong Ma and Frank Dellaert
"""

from math import ceil, log2
from typing import List, Optional, Sequence, Set, Tuple

from gtsam import Ordering, SymbolicFactorGraph  # type: ignore

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.products.clustering import Cluster, Clustering
from gtsfm.products.visibility_graph import VisibilityGraph

logger = logger_utils.get_logger()


class BinaryTreePartitioner(GraphPartitionerBase):
    """Graph partitioner that uses a binary tree to recursively cluster a visibility graph."""

    def __init__(self, max_depth: Optional[int] = None, num_cameras_per_cluster: Optional[int] = None):
        """
        Initialize the BinaryTreePartitioner.

        Args:
            max_depth: Maximum depth of the binary tree; results in at most 2^depth leaf clusters.
            num_cameras_per_cluster: Desired number of cameras per leaf cluster; used to compute max_depth.
        """
        super().__init__(process_name="BinaryTreePartitioner")

        self.max_depth = max_depth
        if max_depth is None:
            if num_cameras_per_cluster is None:
                raise ValueError("Either max_depth or num_cameras_per_cluster must be provided")
            self._num_cameras_per_cluster = num_cameras_per_cluster

    def run(self, graph: VisibilityGraph) -> Clustering:
        """Cluster a visibility graph into a binary tree of clusters."""
        if len(graph) == 0:
            logger.warning("BinaryTreePartitioner: no visibility graph provided for clustering.")
            return Clustering(root=None)

        for i, j in graph:
            if i == j:
                raise ValueError(f"VisibilityGraph contains self-loop ({i}, {j}).")
            if i > j:
                raise ValueError(f"VisibilityGraph contains invalid pair ({i}, {j}): i must be less than j.")

        all_nodes = {node for edge in graph for node in edge}
        num_cameras = len(all_nodes)

        max_depth = self.max_depth
        if max_depth is None:
            max_depth = ceil(log2(max(1, num_cameras) / self._num_cameras_per_cluster))
            max_depth = max(0, max_depth)

        sfg = self._build_symbolic_factor_graph(graph)
        ordering = Ordering.MetisSymbolicFactorGraph(sfg)

        ordered_keys = self._extract_ordered_keys(ordering, all_nodes)
        root_cluster, _, _ = self._build_binary_clustering(
            keys=ordered_keys,
            depth=0,
            max_depth=max_depth,
            graph_edges=graph,
        )
        return Clustering(root=root_cluster)

    def _build_symbolic_factor_graph(self, graph: VisibilityGraph) -> SymbolicFactorGraph:
        """Construct GTSAM graph from visibility graph."""
        sfg = SymbolicFactorGraph()
        for i, j in graph:
            sfg.push_factor(i, j)
        return sfg

    def _extract_ordered_keys(self, ordering: Ordering, all_nodes: Set[int]) -> List[int]:
        """Extract node ordering returned by METIS."""
        ordered_keys: List[int] = []
        seen = set()
        for idx in range(ordering.size()):
            key_value = int(ordering.at(idx))
            if key_value in all_nodes and key_value not in seen:
                ordered_keys.append(key_value)
                seen.add(key_value)

        missing = all_nodes.difference(seen)
        if missing:
            logger.warning("Ordering did not cover all nodes; appending %d missing keys.", len(missing))
            ordered_keys.extend(sorted(missing))
        return ordered_keys

    def _build_binary_clustering(
        self,
        keys: Sequence[int],
        depth: int,
        max_depth: int,
        graph_edges: VisibilityGraph,
    ) -> Tuple[Cluster, Set[int], Set[Tuple[int, int]]]:
        """Recursively build a binary clustering hierarchy.

        Returns:
            A tuple of:
                - Cluster at the current recursion level.
                - Set of keys contained in this cluster and descendants.
                - Set of edges contained in this cluster and descendants.
        """
        key_set = set(keys)

        if depth == max_depth or len(keys) <= 1:
            intra_edges = [(i, j) for i, j in graph_edges if i in key_set and j in key_set]
            cluster = Cluster(keys=frozenset(key_set), edges=intra_edges, children=())
            return cluster, set(cluster.keys), set(intra_edges)

        mid = max(1, len(keys) // 2)
        left_cluster, left_keys, left_edges = self._build_binary_clustering(
            keys=keys[:mid],
            depth=depth + 1,
            max_depth=max_depth,
            graph_edges=graph_edges,
        )
        right_cluster, right_keys, right_edges = self._build_binary_clustering(
            keys=keys[mid:],
            depth=depth + 1,
            max_depth=max_depth,
            graph_edges=graph_edges,
        )

        descendant_keys = left_keys | right_keys
        child_edges = left_edges | right_edges

        cross_edges = [
            (i, j)
            for i, j in graph_edges
            if i in descendant_keys and j in descendant_keys and (i, j) not in child_edges
        ]

        unique_keys = key_set - descendant_keys
        cluster = Cluster(
            keys=frozenset(unique_keys),
            edges=cross_edges,
            children=(left_cluster, right_cluster),
        )

        descendant_keys |= unique_keys
        descendant_edges = child_edges | set(cross_edges)
        return cluster, descendant_keys, descendant_edges
