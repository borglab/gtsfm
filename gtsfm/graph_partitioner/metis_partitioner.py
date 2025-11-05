"""Graph partitioner that constructs a cluster_tree directly from the METIS Bayes tree.

Authors: Frank Dellaert"""

from __future__ import annotations

from dataclasses import dataclass

from gtsam import Ordering, SymbolicBayesTree, SymbolicBayesTreeClique, SymbolicFactorGraph  # type: ignore

from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.visibility_graph import VisibilityGraph, valid_visibility_graph_or_raise
from gtsfm.utils.graph import get_nodes_in_largest_connected_component


@dataclass(frozen=True)
class _SubTreeInfo:
    cluster: ClusterTree
    keys: set[int]
    edges: set[tuple[int, int]]


class MetisPartitioner(GraphPartitionerBase):
    """Graph partitioner that leverages METIS ordering and the symbolic Bayes tree."""

    def __init__(self) -> None:
        super().__init__(process_name="MetisPartitioner")

    def _extract_largest_component_subgraph(self, graph: VisibilityGraph) -> VisibilityGraph:
        nodes_in_largest = set(get_nodes_in_largest_connected_component(graph))
        return [(i, j) for i, j in graph if i in nodes_in_largest and j in nodes_in_largest]

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
            graph = self._extract_largest_component_subgraph(graph)
            roots = self.symbolic_bayes_tree(graph).roots()

            if len(roots) > 1:
                raise ValueError(
                    "MetisPartitioner: VisibilityGraph is disconnected after largest connected component extraction."
                )

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
        """Recursively build the cluster tree, aggressively pruning single-child nodes."""
        keys, frontals, _ = self._clique_key_sets(clique)
        children = [clique[j] for j in range(clique.nrChildren())]

        # Recursively call on children and filter out any pruned (None) results.
        child_results = [res for res in (self._cluster_from_clique(child, graph) for child in children) if res]
        descendant_edges = set.union(*(result.edges for result in child_results)) if child_results else set()

        # Only keep edges that touch at least one frontal variable from this clique.
        candidate_edges = {
            (i, j) for i, j in graph if i in keys and j in keys and (i in frontals or j in frontals or not frontals)
        }
        current_edges = candidate_edges - descendant_edges

        def sorted_edges(edges: set[tuple[int, int]]) -> list[tuple[int, int]]:
            return sorted(edges)

        # Case 1: This node is a leaf in the pruned tree (no valid children remaining).
        if not child_results:
            if not current_edges:
                return None  # Prune empty leaf clusters.
            cluster = ClusterTree(value=sorted_edges(current_edges), children=())
            return _SubTreeInfo(cluster=cluster, keys=keys, edges=current_edges)

        # This node is an internal node in the pruned tree.
        # Calculate its keys, edges, and the edges unique to it.
        subtree_keys = keys | set.union(*(result.keys for result in child_results))
        subtree_edges = current_edges | descendant_edges

        # --- Aggressive Pruning Logic ---
        if len(child_results) == 1:
            # Case 2: Merge with the single child.
            single_child_result = child_results[0]
            # Combine this node's edges with its child's edges.
            merged_cluster = ClusterTree(
                value=sorted_edges(current_edges) + single_child_result.cluster.value,
                children=single_child_result.cluster.children,
            )
            # Pass up the merged cluster, but with the full key/edge sets for this scope.
            return _SubTreeInfo(cluster=merged_cluster, keys=subtree_keys, edges=subtree_edges)
        else:
            # Case 3: Keep this node as a branching point (>1 child).
            cluster = ClusterTree(
                value=sorted_edges(current_edges),
                children=tuple(result.cluster for result in child_results),
            )
            return _SubTreeInfo(cluster=cluster, keys=subtree_keys, edges=subtree_edges)

    def _clique_key_sets(self, clique: SymbolicBayesTreeClique) -> tuple[set[int], set[int], set[int]]:
        conditional = clique.conditional()
        if conditional is not None:
            keys = conditional.keys()
            n_frontals = conditional.nrFrontals()
            frontals = set(int(k) for k in keys[:n_frontals])
            separator = set(int(k) for k in keys[n_frontals:])
            all_keys = frontals | separator
            return all_keys, frontals, separator
        else:
            return set(), set(), set()
