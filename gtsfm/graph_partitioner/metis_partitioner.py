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

    def __init__(self, max_cluster_size: int = 10) -> None:
        super().__init__(process_name="MetisPartitioner")
        self._max_cluster_size = max_cluster_size

    def _extract_largest_component_subgraph(self, graph: VisibilityGraph) -> VisibilityGraph:
        nodes_in_largest = set(get_nodes_in_largest_connected_component(graph))
        return [(i, j) for i, j in graph if i in nodes_in_largest and j in nodes_in_largest]

    def run(self, graph: VisibilityGraph) -> ClusterTree | None:
        """Cluster a visibility graph using the clique structure of the symbolic Bayes tree."""
        result = self._build_cluster_info(graph)
        return result.cluster if result else None

    def _build_cluster_info(self, graph: VisibilityGraph) -> _SubTreeInfo | None:
        if len(graph) == 0:
            return None

        valid_visibility_graph_or_raise(graph)

        bayes_tree = self.symbolic_bayes_tree(graph)
        roots = list(bayes_tree.roots())
        if len(roots) == 0:
            return None
        if len(roots) > 1:
            graph = self._extract_largest_component_subgraph(graph)
            bayes_tree = self.symbolic_bayes_tree(graph)
            roots = list(bayes_tree.roots())
            if len(roots) > 1:
                raise ValueError(
                    "MetisPartitioner: VisibilityGraph is disconnected after largest connected component extraction."
                )

        return self._cluster_from_clique(roots[0], graph)

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
        descendant_edges_pre = set.union(*(result.edges for result in child_results)) if child_results else set()

        # Only keep edges that touch at least one frontal variable from this clique.
        candidate_edges = {
            (i, j) for i, j in graph if i in keys and j in keys and (i in frontals or j in frontals or not frontals)
        }
        current_edges = candidate_edges - descendant_edges_pre

        # Propagate frontal variables into every child cluster so they share overlap with this parent.
        if child_results and frontals:
            distributed_edges: set[tuple[int, int]] = set()
            augmented_children: list[_SubTreeInfo] = []

            for result in child_results:
                extra_edges = {
                    edge
                    for edge in current_edges
                    if (edge[0] in frontals or edge[1] in frontals)
                    and (edge[0] in result.keys or edge[1] in result.keys)
                }
                if extra_edges:
                    merged_edges = set(result.cluster.value) | extra_edges
                    merged_cluster = ClusterTree(
                        value=self._sorted_edges(merged_edges),
                        children=result.cluster.children,
                    )
                    extra_keys = {vertex for edge in extra_edges for vertex in edge}
                    augmented_children.append(
                        _SubTreeInfo(
                            cluster=merged_cluster,
                            keys=result.keys | extra_keys,
                            edges=result.edges | extra_edges,
                        )
                    )
                    distributed_edges |= extra_edges
                else:
                    augmented_children.append(result)

            if distributed_edges:
                current_edges -= distributed_edges
            child_results = augmented_children

        descendant_edges = set.union(*(result.edges for result in child_results)) if child_results else set()
        descendant_keys = set.union(*(result.keys for result in child_results)) if child_results else set()
        subtree_keys = keys | descendant_keys
        subtree_edges = current_edges | descendant_edges

        # Case 1: This node is a leaf in the pruned tree (no valid children remaining).
        if not child_results:
            if not current_edges:
                return None  # Prune empty leaf clusters.
            cluster = ClusterTree(value=self._sorted_edges(current_edges), children=())
            info = _SubTreeInfo(cluster=cluster, keys=subtree_keys, edges=subtree_edges)
            return self._maybe_refine_large_cluster(info, graph, frontals)

        # This node is an internal node in the pruned tree.
        # Calculate its keys, edges, and the edges unique to it.

        # --- Aggressive Pruning Logic ---
        if len(child_results) == 1:
            # Case 2: Merge with the single child.
            single_child_result = child_results[0]
            # Combine this node's edges with its child's edges.
            merged_cluster = ClusterTree(
                value=self._sorted_edges(current_edges) + single_child_result.cluster.value,
                children=single_child_result.cluster.children,
            )
            # Pass up the merged cluster, but with the full key/edge sets for this scope.
            info = _SubTreeInfo(cluster=merged_cluster, keys=subtree_keys, edges=subtree_edges)
            return self._maybe_refine_large_cluster(info, graph, frontals)
        else:
            # Case 3: Keep this node as a branching point (>1 child).
            cluster = ClusterTree(
                value=self._sorted_edges(current_edges),
                children=tuple(result.cluster for result in child_results),
            )
            info = _SubTreeInfo(cluster=cluster, keys=subtree_keys, edges=subtree_edges)
            return self._maybe_refine_large_cluster(info, graph, frontals)

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

    def _maybe_refine_large_cluster(
        self,
        info: _SubTreeInfo,
        graph: VisibilityGraph,
        frontals: set[int] | None,
    ) -> _SubTreeInfo:
        """Subdivide leaf clusters that exceed the desired max cluster size.

        When subdividing, each child cluster receives the parent frontal keys to keep pose anchors shared.
        """
        if len(info.keys) <= self._max_cluster_size or not info.cluster.is_leaf():
            return info

        anchor_keys = set(frontals or set())

        subgraph_edges = self._induced_subgraph(info.keys, graph)
        subgraph_graph = list(subgraph_edges)
        anchor_edges: set[tuple[int, int]] = {
            edge for edge in subgraph_edges if (edge[0] in anchor_keys) or (edge[1] in anchor_keys)
        }

        recursive_result = self._build_cluster_info(subgraph_graph)
        if recursive_result is not None:
            refined_cluster = self._propagate_anchor_edges(recursive_result.cluster, anchor_edges)
            if not self._clusters_equal(refined_cluster, info.cluster):
                return _SubTreeInfo(cluster=refined_cluster, keys=info.keys, edges=subgraph_edges)

        ordered_keys = self._metis_order_keys(subgraph_edges, info.keys)
        if anchor_keys:
            ordered_keys = [k for k in ordered_keys if k not in anchor_keys]
        key_groups = self._chunk_ordered_keys(ordered_keys)

        if len(key_groups) <= 1:
            return info

        child_clusters: list[ClusterTree] = []
        child_edge_union: set[tuple[int, int]] = set()

        for group in key_groups:
            group_set = set(group)
            group_edges = {
                (i, j)
                for (i, j) in subgraph_edges
                if i in group_set and j in group_set
            }
            child_clusters.append(ClusterTree(value=self._sorted_edges(group_edges), children=()))
            child_edge_union |= group_edges

        cross_edges = subgraph_edges - child_edge_union
        refined_cluster = ClusterTree(value=self._sorted_edges(cross_edges), children=tuple(child_clusters))
        refined_cluster = self._propagate_anchor_edges(refined_cluster, anchor_edges)
        return _SubTreeInfo(cluster=refined_cluster, keys=info.keys, edges=subgraph_edges)

    def _metis_order_keys(
        self, subgraph_edges: set[tuple[int, int]], default_keys: set[int]
    ) -> list[int]:
        """Return METIS ordering for a subgraph, with graceful fallbacks."""
        if not subgraph_edges:
            return sorted(default_keys)

        ordering = Ordering.MetisSymbolicFactorGraph(self._symbolic_factor_graph(list(subgraph_edges)))
        keys: list[int] = []

        if hasattr(ordering, "keys"):
            keys = [int(k) for k in ordering.keys()]
        elif hasattr(ordering, "keyVector"):
            keys = [int(k) for k in ordering.keyVector()]
        elif hasattr(ordering, "key"):
            size = ordering.size()
            keys = [int(ordering.key(i)) for i in range(size)]

        if not keys:
            # Final fallback: rely on provided key set.
            return sorted(default_keys)

        # Ensure all requested keys are covered; fill any missing ones deterministically.
        remaining = list(sorted(default_keys - set(keys)))
        keys.extend(remaining)
        return keys

    def _chunk_ordered_keys(self, ordered_keys: list[int]) -> list[list[int]]:
        """Split an ordered key list into contiguous groups respecting the max cluster size."""
        return [
            ordered_keys[idx : idx + self._max_cluster_size]
            for idx in range(0, len(ordered_keys), self._max_cluster_size)
            if ordered_keys[idx : idx + self._max_cluster_size]
        ]

    @staticmethod
    def _induced_subgraph(keys: set[int], graph: VisibilityGraph) -> set[tuple[int, int]]:
        """Return all edges whose endpoints lie entirely inside the provided key set."""
        return {(i, j) for i, j in graph if i in keys and j in keys}

    @staticmethod
    def _sorted_edges(edges: set[tuple[int, int]]) -> list[tuple[int, int]]:
        """Consistently order edges when storing them in ClusterTree nodes."""
        return sorted(edges)

    def _clusters_equal(self, lhs: ClusterTree, rhs: ClusterTree) -> bool:
        if set(lhs.value) != set(rhs.value):
            return False
        if len(lhs.children) != len(rhs.children):
            return False
        return all(self._clusters_equal(lc, rc) for lc, rc in zip(lhs.children, rhs.children))

    def _propagate_anchor_edges(
        self,
        cluster: ClusterTree,
        anchor_edges: set[tuple[int, int]],
    ) -> ClusterTree:
        if not anchor_edges:
            return cluster

        merged_value = self._sorted_edges(set(cluster.value) | anchor_edges)
        if not cluster.children:
            return ClusterTree(value=merged_value, children=())

        new_children = tuple(self._propagate_anchor_edges(child, anchor_edges) for child in cluster.children)
        return ClusterTree(value=merged_value, children=new_children)
