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

    def __init__(self, min_cameras_to_partition: int | None = None, max_cameras: int | None = None) -> None:
        super().__init__(process_name="MetisPartitioner")
        if min_cameras_to_partition is not None and min_cameras_to_partition < 1:
            raise ValueError("min_cameras_to_partition must be >= 1 when provided.")
        if max_cameras is not None and max_cameras < 1:
            raise ValueError("max_cameras must be >= 1 when provided.")
        if max_cameras is not None and min_cameras_to_partition is not None and max_cameras < min_cameras_to_partition:
            raise ValueError("max_cameras must be >= min_cameras_to_partition when provided.")
        self._min_cameras_to_partition = min_cameras_to_partition
        self._max_cameras = max_cameras

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

    @staticmethod
    def _min_key(keys: set[int]) -> int:
        return min(keys) if keys else 10**18

    @staticmethod
    def _count_cross_edges(keys_a: set[int], keys_b: set[int], graph: VisibilityGraph) -> int:
        return sum(1 for i, j in graph if (i in keys_a and j in keys_b) or (i in keys_b and j in keys_a))

    def _are_mergeable(self, keys_a: set[int], keys_b: set[int], graph: VisibilityGraph) -> bool:
        if len(keys_a & keys_b) > 0:
            return True
        return self._count_cross_edges(keys_a, keys_b, graph) > 0

    def _merge_subtrees(self, a: _SubTreeInfo, b: _SubTreeInfo) -> _SubTreeInfo:
        merged_keys = a.keys | b.keys
        merged_edges = a.edges | b.edges
        # Flatten one level when merging siblings to keep tree depth stable.
        merged_cluster = ClusterTree(
            value=sorted(set(a.cluster.value) | set(b.cluster.value)),
            children=a.cluster.children + b.cluster.children,
        )
        return _SubTreeInfo(cluster=merged_cluster, keys=merged_keys, edges=merged_edges)

    def _merge_small_children_at_level(
        self, children: list[_SubTreeInfo], graph: VisibilityGraph
    ) -> list[_SubTreeInfo]:
        min_cameras = self._min_cameras_to_partition
        if min_cameras is None or len(children) < 2:
            return children

        work = list(children)
        max_cameras = self._max_cameras

        while True:
            small_indices = [idx for idx, child in enumerate(work) if len(child.keys) < min_cameras]
            if not small_indices:
                break

            # Process the smallest / hardest-to-place child first.
            target_idx = min(
                small_indices,
                key=lambda idx: (len(work[idx].keys), self._min_key(work[idx].keys), idx),
            )
            target = work[target_idx]

            candidates: list[int] = []
            for idx, child in enumerate(work):
                if idx == target_idx:
                    continue
                merged_size = len(target.keys | child.keys)
                if max_cameras is not None and merged_size > max_cameras:
                    continue
                if self._are_mergeable(target.keys, child.keys, graph):
                    candidates.append(idx)

            if not candidates:
                break

            # Prefer merging with another small sibling whenever possible.
            small_candidates = [idx for idx in candidates if len(work[idx].keys) < min_cameras]
            if small_candidates:
                candidates = small_candidates

            def candidate_rank(idx: int) -> tuple[int, int, int, int, int, int]:
                candidate = work[idx]
                merged_keys = target.keys | candidate.keys
                merged_size = len(merged_keys)
                reaches_threshold = 1 if merged_size >= min_cameras else 0
                overshoot = merged_size - min_cameras if merged_size >= min_cameras else 10**18
                shared_keys = len(target.keys & candidate.keys)
                cross_edges = self._count_cross_edges(target.keys, candidate.keys, graph)
                return (
                    -reaches_threshold,
                    overshoot,
                    -shared_keys,
                    -cross_edges,
                    merged_size,
                    self._min_key(candidate.keys),
                )

            partner_idx = min(candidates, key=candidate_rank)
            partner = work[partner_idx]
            merged = self._merge_subtrees(target, partner)

            i, j = sorted([target_idx, partner_idx], reverse=True)
            work.pop(i)
            work.pop(j)
            work.append(merged)

        return work

    def _cluster_from_clique(self, clique: SymbolicBayesTreeClique, graph: VisibilityGraph) -> _SubTreeInfo | None:
        """Recursively build the cluster tree, aggressively pruning single-child nodes."""
        keys, frontals, _ = self._clique_key_sets(clique)
        children = [clique[j] for j in range(clique.nrChildren())]

        # Recursively call on children and filter out any pruned (None) results.
        child_results = [res for res in (self._cluster_from_clique(child, graph) for child in children) if res]
        child_results = self._merge_small_children_at_level(child_results, graph)
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

        # Collapse small subtrees into leaves.
        if self._max_cameras is not None and len(subtree_keys) <= self._max_cameras:
            collapsed_cluster = ClusterTree(value=sorted_edges(subtree_edges), children=())
            return _SubTreeInfo(cluster=collapsed_cluster, keys=subtree_keys, edges=subtree_edges)

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
