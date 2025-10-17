"""Graph partitioner that constructs a cluster_tree directly from the METIS Bayes tree.

Authors: Frank Dellaert"""

from __future__ import annotations

from dataclasses import dataclass

from gtsam import Ordering, SymbolicBayesTree, SymbolicBayesTreeClique, SymbolicFactorGraph  # type: ignore

from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.visibility_graph import VisibilityGraph, valid_visibility_graph_or_raise


@dataclass(frozen=True)
class _CliqueClusterResult:
    cluster: ClusterTree
    keys: set[int]
    edges: set[tuple[int, int]]


class MetisPartitioner(GraphPartitionerBase):
    """Graph partitioner that leverages METIS ordering and the symbolic Bayes tree."""

    def __init__(self) -> None:
        super().__init__(process_name="MetisPartitioner")

    def run(self, graph: VisibilityGraph) -> ClusterTree | None:
        """Cluster a visibility graph using the clique structure of the symbolic Bayes tree."""
        if len(graph) == 0:
            return None

        valid_visibility_graph_or_raise(graph)

        bayes_tree = self.symbolic_bayes_tree(graph)
        roots: list = bayes_tree.roots()
        if len(roots) == 0:
            return None
        if len(roots) > 1:
            raise ValueError("MetisPartitioner: VisibilityGraph is disconnected.")
        root_result = self._cluster_from_clique(roots[0], graph)
        return root_result.cluster

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

    def _cluster_from_clique(self, clique: SymbolicBayesTreeClique, graph: VisibilityGraph) -> _CliqueClusterResult:
        frontals, _ = self._clique_key_sets(clique)
        children = [clique[j] for j in range(clique.nrChildren())]

        if not children:
            # Create a leaf cluster.
            edges = [(i, j) for i, j in graph if i in frontals and j in frontals]
            cluster = ClusterTree(value=edges, children=())
            return _CliqueClusterResult(cluster=cluster, keys=set(frontals), edges=set(edges))

        child_results = [self._cluster_from_clique(child, graph) for child in children]

        descendant_keys: set[int] = set.union(*(result.keys for result in child_results))
        descendant_edges: set[tuple[int, int]] = set.union(*(result.edges for result in child_results))

        subtree_keys = descendant_keys | frontals
        subtree_edges = {(i, j) for i, j in graph if i in subtree_keys and j in subtree_keys}

        cluster = ClusterTree(
            value=list(subtree_edges - descendant_edges),
            children=tuple(result.cluster for result in child_results),
        )
        return _CliqueClusterResult(cluster=cluster, keys=subtree_keys, edges=subtree_edges)

    def _clique_key_sets(self, clique: SymbolicBayesTreeClique) -> tuple[set[int], set[int]]:
        conditional = clique.conditional()
        if conditional is not None:
            keys = conditional.keys()
            n_frontals = conditional.nrFrontals()
            frontals = set(int(k) for k in keys[:n_frontals])
            separator = set(int(k) for k in keys[n_frontals:])
            return frontals, separator
        else:
            return set(), set()
