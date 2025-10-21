"""Graph partitioner that constructs a cluster_tree directly from the METIS Bayes tree.

Authors: Frank Dellaert"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from gtsam import Ordering, SymbolicBayesTree, SymbolicBayesTreeClique, SymbolicFactorGraph  # type: ignore

from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.visibility_graph import VisibilityGraph, valid_visibility_graph_or_raise


@dataclass(frozen=True)
class _SubTreeInfo:
    cluster: ClusterTree
    keys: set[int]
    edges: set[tuple[int, int]]


class MetisPartitioner(GraphPartitionerBase):
    """Graph partitioner that leverages METIS ordering and the symbolic Bayes tree."""

    def __init__(self) -> None:
        super().__init__(process_name="MetisPartitioner")

    def run(self, graph: VisibilityGraph, write_csv: bool = False) -> ClusterTree | None:
        """Cluster a visibility graph using the clique structure of the symbolic Bayes tree."""
        if len(graph) == 0:
            return None

        valid_visibility_graph_or_raise(graph)

        # Optionally, write the visibility graph to a CSV for inspection.
        if write_csv:
            with open("visibility_graph.csv", "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["i", "j"])
                for i, j in graph:
                    writer.writerow([i, j])

        bayes_tree = self.symbolic_bayes_tree(graph)
        roots: list = bayes_tree.roots()
        if len(roots) == 0:
            return None
        if len(roots) > 1:
            raise ValueError("MetisPartitioner: VisibilityGraph is disconnected.")
        root_result = self._cluster_from_clique(roots[0], graph)
        return root_result.cluster if root_result else None

    def symbolic_bayes_tree(self, graph: VisibilityGraph) -> SymbolicBayesTree:
        """Helper to build the Bayes tree from the visibility graph."""
        sfg = self._symbolic_factor_graph(graph)
        ordering = Ordering.MetisSymbolicFactorGraph(sfg)
        return sfg.eliminateMultifrontal(ordering)

    def _symbolic_factor_graph(self, graph: VisibilityGraph) -> SymbolicFactorGraph:
        sfg = SymbolicFactorGraph()
        for i, j in graph:
            sfg.push_factor(i, j)
        return sfg

    def _cluster_from_clique(self, clique: SymbolicBayesTreeClique, graph: VisibilityGraph) -> _SubTreeInfo | None:
        """Recursively build the cluster tree, pruning redundant nodes."""
        keys, frontals, _ = self._clique_key_sets(clique)
        children = [clique[j] for j in range(clique.nrChildren())]

        # Recursively call on children and filter out any pruned (None) results.
        child_subtrees = [
            result for result in (self._cluster_from_clique(child, graph) for child in children) if result
        ]

        if not child_subtrees:
            # Base case: This is a leaf clique.
            edges = {(i, j) for i, j in graph if i in keys and j in keys}
            if not edges:
                return None  # Prune empty leaf clusters.
            cluster = ClusterTree(value=list(edges), children=())
            return _SubTreeInfo(cluster=cluster, keys=keys, edges=edges)

        # The keys for this subtree are the union of the frontal keys and all descendant keys.
        subtree_keys = frontals | set.union(*(result.keys for result in child_subtrees))

        # The edges for this subtree are all edges in the visibility graph that connect subtree keys.
        subtree_edges = {(i, j) for i, j in graph if i in subtree_keys and j in subtree_keys}

        # The cluster at the top of this subtree are those not covered by the descendants.
        descendant_edges: set[tuple[int, int]] = set.union(*(result.edges for result in child_subtrees))

        current_edges = subtree_edges - descendant_edges
        child_clusters = tuple(result.cluster for result in child_subtrees)

        # --- Pruning Logic ---
        if not current_edges:
            # If this node has no unique edges, it may be redundant.
            if len(child_clusters) == 1:
                # Node is redundant. Collapse it by returning its single child's info.
                # The returned info uses the child's cluster but the parent's full key/edge sets.
                return _SubTreeInfo(cluster=child_clusters[0], keys=subtree_keys, edges=subtree_edges)
            elif len(child_clusters) == 0:
                # This was an internal node, but all its children were pruned.
                return None

        # This node is not redundant, create a new cluster.
        # It's kept if it has its own edges or if it's a necessary join for multiple children.
        cluster = ClusterTree(value=list(current_edges), children=child_clusters)
        return _SubTreeInfo(cluster=cluster, keys=subtree_keys, edges=subtree_edges)

    def _clique_key_sets(self, clique: SymbolicBayesTreeClique) -> tuple[set[int], set[int], set[int]]:
        conditional = clique.conditional()
        if conditional is not None:
            keys = conditional.keys()
            n_frontals = conditional.nrFrontals()
            frontals = set(int(k) for k in keys[:n_frontals])
            separator = set(int(k) for k in keys[n_frontals:])
            return set(keys), frontals, separator
        else:
            return set(), set(), set()
