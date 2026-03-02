"""Graph partitioner that constructs a cluster_tree directly from the METIS Bayes tree.

Authors: Frank Dellaert"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import cast

import gtsfm.utils.logger as logger_utils
from gtsam import Ordering, SymbolicBayesTree, SymbolicBayesTreeClique, SymbolicFactorGraph  # type: ignore

from gtsfm.graph_partitioner import partition_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.visibility_graph import VisibilityGraph, valid_visibility_graph_or_raise, visibility_graph_keys
from gtsfm.utils.graph import get_nodes_in_largest_connected_component

logger = logger_utils.get_logger()


@dataclass(frozen=True)
class _SubTreeInfo:
    cluster: ClusterTree
    keys: set[int]
    edges: set[tuple[int, int]]


class MetisPartitioner(GraphPartitionerBase):
    """Graph partitioner that leverages METIS ordering and the symbolic Bayes tree."""

    def __init__(
        self,
        min_cameras_to_partition: int | None = None,
        max_cameras: int | None = None,
        min_child_overlap_for_split: int = 2,
        min_parent_overlap_for_split: int = 2,
        split_oversized_nodes: bool = False,
    ) -> None:
        super().__init__(process_name="MetisPartitioner")
        if min_cameras_to_partition is not None and min_cameras_to_partition < 1:
            raise ValueError("min_cameras_to_partition must be >= 1 when provided.")
        if max_cameras is not None and max_cameras < 1:
            raise ValueError("max_cameras must be >= 1 when provided.")
        if min_child_overlap_for_split < 1:
            raise ValueError("min_child_overlap_for_split must be >= 1.")
        if min_parent_overlap_for_split < 1:
            raise ValueError("min_parent_overlap_for_split must be >= 1.")
        if max_cameras is not None and min_cameras_to_partition is not None and max_cameras < min_cameras_to_partition:
            raise ValueError("max_cameras must be >= min_cameras_to_partition when provided.")
        self._min_cameras_to_partition = min_cameras_to_partition
        self._max_cameras = max_cameras
        self._min_child_overlap_for_split = min_child_overlap_for_split
        self._min_parent_overlap_for_split = min_parent_overlap_for_split
        self._split_oversized_nodes = split_oversized_nodes

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
        if root_result is None:
            return None
        logger.info(
            "MetisPartitioner: initial root cameras=%d, split_oversized_nodes=%s, max_cameras=%s",
            len(root_result.keys),
            self._split_oversized_nodes,
            self._max_cameras,
        )
        if not self._split_oversized_nodes:
            return root_result.cluster
        split_root = self._split_oversized_tree(root_result.cluster, graph)
        logger.info("MetisPartitioner: post-split root cameras=%d", len(split_root.all_keys()))
        return split_root

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

    def _are_mergeable(self, keys_a: set[int], keys_b: set[int], graph: VisibilityGraph) -> bool:
        if len(keys_a & keys_b) > 0:
            return True
        return partition_utils.count_cross_edges(keys_a, keys_b, graph) > 0

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
                key=lambda idx: (len(work[idx].keys), partition_utils.min_key(work[idx].keys), idx),
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

            def candidate_rank(idx: int) -> tuple[int, ...]:
                candidate = work[idx]
                merged_keys = target.keys | candidate.keys
                merged_size = len(merged_keys)
                reaches_threshold = 1 if merged_size >= min_cameras else 0
                overshoot = merged_size - min_cameras if merged_size >= min_cameras else 10**18
                shared_keys = len(target.keys & candidate.keys)
                cross_edges = partition_utils.count_cross_edges(target.keys, candidate.keys, graph)
                # shared_keys first, cross_edges next; merged_size / reaches_threshold for tie-breaks.
                return (
                    -shared_keys,
                    -cross_edges,
                    merged_size,
                    -reaches_threshold,
                    overshoot,
                    partition_utils.min_key(candidate.keys),
                )

            partner_idx = min(candidates, key=candidate_rank)
            partner = work[partner_idx]
            merged = self._merge_subtrees(target, partner)

            i, j = sorted([target_idx, partner_idx], reverse=True)
            work.pop(i)
            work.pop(j)
            work.append(merged)

        return work

    @staticmethod
    def _local_keys(node: ClusterTree) -> set[int]:
        return set(visibility_graph_keys(node.value))

    def _ensure_parent_overlap(self, bin_keys: list[set[int]], parent_local_keys: set[int] | None) -> bool:
        if parent_local_keys is None or len(parent_local_keys) == 0 or self._max_cameras is None:
            return True
        for keys in bin_keys:
            overlap = len(keys & parent_local_keys)
            if overlap >= self._min_parent_overlap_for_split:
                continue
            need = self._min_parent_overlap_for_split - overlap
            candidates = sorted((parent_local_keys - keys))
            if len(candidates) < need or len(keys) + need > self._max_cameras:
                return False
            keys.update(candidates[:need])
        return True

    def _assign_children_to_bins(
        self,
        children: list[ClusterTree],
        child_keysets: list[set[int]],
        bin_keys: list[set[int]],
        graph: VisibilityGraph,
    ) -> tuple[list[list[ClusterTree]], bool]:
        """Assign each child to a bin. May mutate bin_keys for augmentation. Returns (assigned_children, feasible)."""
        k = len(bin_keys)
        assigned_children: list[list[ClusterTree]] = [[] for _ in range(k)]
        child_order = sorted(
            range(len(children)),
            key=lambda idx: (
                max(len(child_keysets[idx] & b) for b in bin_keys),
                len(child_keysets[idx]),
                partition_utils.min_key(child_keysets[idx]),
                idx,
            ),
        )
        for child_idx in child_order:
            child = children[child_idx]
            child_keys = child_keysets[child_idx]
            overlaps = [len(child_keys & keys) for keys in bin_keys]
            candidate_bins = [
                idx for idx, overlap in enumerate(overlaps) if overlap >= self._min_child_overlap_for_split
            ]
            if not candidate_bins:
                augmentation_options: list[tuple[int, int, int, list[int]]] = []
                for idx, keys in enumerate(bin_keys):
                    missing = self._min_child_overlap_for_split - overlaps[idx]
                    if missing <= 0:
                        continue
                    available = sorted((child_keys - keys))
                    if len(available) < missing:
                        continue
                    if len(keys) + missing > self._max_cameras:
                        continue
                    augmentation_options.append((missing, len(keys), idx, available[:missing]))
                if not augmentation_options:
                    return assigned_children, False
                _, _, chosen_idx, to_add = min(augmentation_options, key=lambda item: (item[0], item[1], item[2]))
                bin_keys[chosen_idx].update(to_add)
                overlaps[chosen_idx] = len(child_keys & bin_keys[chosen_idx])
                candidate_bins = [chosen_idx]

            best_idx = max(
                candidate_bins,
                key=lambda idx: (
                    overlaps[idx],
                    partition_utils.count_cross_edges(child_keys, bin_keys[idx], graph),
                    -len(bin_keys[idx]),
                    -idx,
                ),
            )
            if len(child_keys & bin_keys[best_idx]) < self._min_child_overlap_for_split:
                return assigned_children, False
            assigned_children[best_idx].append(child)
        return assigned_children, True

    def _assign_edges_to_bins(
        self,
        node: ClusterTree,
        bin_keys: list[set[int]],
    ) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
        """Assign node edges to bins. May mutate bin_keys. Returns (bin_edges, dropped_edge_list)."""
        k = len(bin_keys)
        bin_edges: list[list[tuple[int, int]]] = [[] for _ in range(k)]
        dropped_edge_list: list[tuple[int, int]] = []
        for i, j in sorted(set(node.value)):
            both = [idx for idx, keys in enumerate(bin_keys) if i in keys and j in keys]
            if both:
                best_idx = min(both, key=lambda idx: (len(bin_edges[idx]), idx))
                bin_edges[best_idx].append((i, j))
                continue

            one = [idx for idx, keys in enumerate(bin_keys) if i in keys or j in keys]
            extended = False
            for idx in one:
                keys = bin_keys[idx]
                add_keys = []
                if i not in keys:
                    add_keys.append(i)
                if j not in keys:
                    add_keys.append(j)
                if len(keys) + len(add_keys) <= self._max_cameras:
                    keys.update(add_keys)
                    bin_edges[idx].append((i, j))
                    extended = True
                    break
            if not extended:
                dropped_edge_list.append((i, j))
        return bin_edges, dropped_edge_list

    def _materialize_split_nodes(
        self,
        bin_edges: list[list[tuple[int, int]]],
        assigned_children: list[list[ClusterTree]],
        child_keys_by_id: dict[int, set[int]],
        parent_local_keys: set[int] | None,
        is_root: bool,
    ) -> tuple[list[ClusterTree], bool]:
        """Build split nodes and validate constraints. Returns (split_nodes, feasible)."""
        k = len(bin_edges)
        split_nodes: list[ClusterTree] = []
        for idx in range(k):
            if len(bin_edges[idx]) == 0 and len(assigned_children[idx]) == 0:
                continue
            split_node = ClusterTree(value=sorted(set(bin_edges[idx])), children=tuple(assigned_children[idx]))
            split_local_keys = self._local_keys(split_node)
            if len(split_local_keys) > self._max_cameras:
                return split_nodes, False
            if not is_root and parent_local_keys is not None:
                if len(split_local_keys & parent_local_keys) < self._min_parent_overlap_for_split:
                    return split_nodes, False
            for child in assigned_children[idx]:
                child_keys = child_keys_by_id[id(child)]
                if len(child_keys & split_local_keys) < self._min_child_overlap_for_split:
                    return split_nodes, False
            split_nodes.append(split_node)
        return split_nodes, len(split_nodes) >= 2

    def _build_root_split_node(
        self,
        split_nodes: list[ClusterTree],
        dropped_edge_list: list[tuple[int, int]],
        graph: VisibilityGraph,
    ) -> ClusterTree | None:
        """Build synthetic root with overlap keys. Returns root node or None if infeasible."""
        required = self._min_parent_overlap_for_split
        split_local_keysets = [self._local_keys(split_node) for split_node in split_nodes]
        if any(len(keys) < required for keys in split_local_keysets):
            return None

        endpoint_score: dict[int, int] = {}
        for i, j in dropped_edge_list:
            endpoint_score[i] = endpoint_score.get(i, 0) + 1
            endpoint_score[j] = endpoint_score.get(j, 0) + 1

        root_keys: set[int] = set()
        impact_order = sorted(endpoint_score.keys(), key=lambda cam: (-endpoint_score[cam], cam))
        for cam in impact_order:
            if len(root_keys) >= self._max_cameras:
                break
            root_keys.add(cam)

        for keys in split_local_keysets:
            overlap = len(keys & root_keys)
            if overlap >= required:
                continue
            need = required - overlap
            candidates = sorted((keys - root_keys), key=lambda cam: (-endpoint_score.get(cam, 0), cam))
            if len(candidates) < need or len(root_keys) + need > self._max_cameras:
                return None
            root_keys.update(candidates[:need])

        if any(len(keys & root_keys) < required for keys in split_local_keysets):
            return None

        root_edges = partition_utils.build_edges_for_keyset(root_keys, graph)
        absorbed_dropped_edges = [e for e in dropped_edge_list if e[0] in root_keys and e[1] in root_keys]
        root_edges = sorted(set(root_edges) | set(absorbed_dropped_edges))
        root_local = set(visibility_graph_keys(root_edges))
        if len(root_local) > self._max_cameras:
            return None
        if any(len(keys & root_local) < required for keys in split_local_keysets):
            return None
        return ClusterTree(value=root_edges, children=tuple(split_nodes))

    def _attempt_split_node(
        self, node: ClusterTree, graph: VisibilityGraph, parent_local_keys: set[int] | None, is_root: bool
    ) -> list[ClusterTree] | None:
        """Attempt local-key-based splitting with minimum number of split nodes."""
        if self._max_cameras is None:
            return None

        local_keys = sorted(self._local_keys(node))
        if len(local_keys) <= self._max_cameras:
            return None

        children = [cast(ClusterTree, c) for c in node.children]
        child_keysets = [set(child.all_keys()) for child in children]
        child_keys_by_id = {id(child): child_keysets[idx] for idx, child in enumerate(children)}

        k_min = max(2, ceil(len(local_keys) / self._max_cameras))
        k_max = len(local_keys)

        logger.info(
            "MetisPartitioner: attempting split on node with local_cameras=%d, children=%d, k_min=%d",
            len(local_keys),
            len(children),
            k_min,
        )

        for k in range(k_min, k_max + 1):
            bin_keys = partition_utils.partition_local_keys(local_keys, k)
            if not self._ensure_parent_overlap(bin_keys, parent_local_keys if not is_root else None):
                continue

            assigned_children, feasible = self._assign_children_to_bins(children, child_keysets, bin_keys, graph)
            if not feasible:
                continue

            bin_edges, dropped_edge_list = self._assign_edges_to_bins(node, bin_keys)

            split_nodes, feasible = self._materialize_split_nodes(
                bin_edges, assigned_children, child_keys_by_id, parent_local_keys, is_root
            )
            if not feasible:
                continue

            if is_root:
                root_node = self._build_root_split_node(split_nodes, dropped_edge_list, graph)
                if root_node is None:
                    continue
                candidate_nodes = [root_node]
                root_local = self._local_keys(root_node)
                absorbed = sum(1 for e in dropped_edge_list if e[0] in root_local and e[1] in root_local)
                effective_dropped_edges = len(dropped_edge_list) - absorbed
            else:
                candidate_nodes = split_nodes
                effective_dropped_edges = len(dropped_edge_list)

            split_children = candidate_nodes[0].children if is_root else candidate_nodes
            logger.info(
                "MetisPartitioner: split accepted with k=%d (local sizes=%s, dropped_edges=%d)",
                len(split_children),
                [len(self._local_keys(split_node)) for split_node in split_children],
                effective_dropped_edges,
            )
            return candidate_nodes

        logger.info("MetisPartitioner: no feasible split satisfies overlap constraints, keeping node unchanged")
        return None

    def _augment_child_local_overlap(
        self,
        child: ClusterTree,
        parent_local_keys: set[int],
        adjacency: dict[int, set[int]],
    ) -> ClusterTree:
        """Augment child local keys to satisfy minimum overlap with parent local keys."""
        if self._min_child_overlap_for_split <= 0 or len(parent_local_keys) == 0:
            return child

        child_local = self._local_keys(child)
        overlap = child_local & parent_local_keys
        required = self._min_child_overlap_for_split
        if len(overlap) >= required:
            return child

        needed = required - len(overlap)
        candidates = sorted(
            list(parent_local_keys - child_local),
            key=lambda cam: (
                -sum(1 for nbr in adjacency.get(cam, set()) if nbr in parent_local_keys),
                cam,
            ),
        )
        if not candidates:
            return child

        existing_edges = set(child.value)
        extra_edges: set[tuple[int, int]] = set()
        for cam in candidates:
            if needed <= 0:
                break
            # Prefer connecting to existing child local keys, then parent local keys.
            anchor = None
            neighbors = adjacency.get(cam, set())
            child_anchors = sorted(child_local & neighbors)
            if child_anchors:
                anchor = child_anchors[0]
            else:
                parent_anchors = sorted(parent_local_keys & neighbors)
                if parent_anchors:
                    anchor = parent_anchors[0]
                elif child_local:
                    anchor = min(child_local)
                else:
                    parent_fallback = sorted(parent_local_keys - {cam})
                    anchor = parent_fallback[0] if parent_fallback else None

            if anchor is None or anchor == cam:
                continue
            edge = partition_utils.canonical_edge(cam, anchor)
            extra_edges.add(edge)
            child_local.update([cam, anchor])
            needed -= 1

        if not extra_edges:
            return child

        return ClusterTree(value=sorted(existing_edges | extra_edges), children=child.children)

    def _preprocess_min_child_overlap(self, node: ClusterTree, adjacency: dict[int, set[int]]) -> ClusterTree:
        """Top-down pass to increase child local overlap with parent local keys."""
        parent_local = self._local_keys(node)
        updated_children: list[ClusterTree] = []
        for raw_child in node.children:
            child = cast(ClusterTree, raw_child)
            augmented_child = self._augment_child_local_overlap(child, parent_local, adjacency)
            updated_children.append(self._preprocess_min_child_overlap(augmented_child, adjacency))
        return ClusterTree(value=list(node.value), children=tuple(updated_children))

    def _split_node_recursive(
        self, node: ClusterTree, graph: VisibilityGraph, parent_local_keys: set[int] | None, is_root: bool
    ) -> list[ClusterTree]:
        """Top-down splitter: split current node first, then recurse into children."""
        rebuilt = ClusterTree(value=list(node.value), children=tuple(cast(ClusterTree, c) for c in node.children))
        split_nodes = self._attempt_split_node(rebuilt, graph, parent_local_keys, is_root)
        current_nodes = split_nodes if split_nodes is not None else [rebuilt]

        refined_nodes: list[ClusterTree] = []
        for current in current_nodes:
            next_parent_local_keys = self._local_keys(current)
            refined_children: list[ClusterTree] = []
            for child in current.children:
                child_cluster = cast(ClusterTree, child)
                refined_children.extend(
                    self._split_node_recursive(
                        child_cluster,
                        graph=graph,
                        parent_local_keys=next_parent_local_keys,
                        is_root=False,
                    )
                )
            refined_nodes.append(ClusterTree(value=list(current.value), children=tuple(refined_children)))
        return refined_nodes

    def _split_oversized_tree(self, root: ClusterTree, graph: VisibilityGraph) -> ClusterTree:
        """Split oversized nodes after initial METIS tree construction."""
        adjacency = partition_utils.graph_adjacency(graph)
        preprocessed = self._preprocess_min_child_overlap(root, adjacency=adjacency)
        split_roots = self._split_node_recursive(preprocessed, graph=graph, parent_local_keys=None, is_root=True)
        return split_roots[0]

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

        # Collapse subtrees that fit within max_cameras into leaves.
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
