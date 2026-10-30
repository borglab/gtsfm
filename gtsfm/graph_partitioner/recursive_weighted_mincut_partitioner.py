"""Binary cluster tree from recursive NetworkX weighted min-cuts.

Same recursive structure and boundary-camera overlap as
``RecursiveMetisPartitioner``, but each bipartition is a global weighted
min-cut (Stoer–Wagner) instead of METIS. Edge weight defaults to 1; when
inlier counts are supplied they are used as capacities so weakly matched
pairs are preferred cut locations.

Stoer–Wagner does not enforce balance. ``max_part_fraction`` rejects a cut
whose larger side exceeds that share of the cameras, falling back to a
midpoint key split (same fallback METIS uses when it cannot cut).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import networkx as nx

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.graph_partitioner.partition_utils import canonical_edge
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.visibility_graph import VisibilityGraph, valid_visibility_graph_or_raise, visibility_graph_keys

logger = logger_utils.get_logger()


@dataclass
class _PartitionNode:
    """Disjoint camera set produced by one recursive bipartition."""

    keys: tuple[int, ...]
    left: "_PartitionNode | None" = None
    right: "_PartitionNode | None" = None
    borrowed: set[int] = field(default_factory=set)

    def native_keys(self) -> set[int]:
        if self.left is None or self.right is None:
            return set(self.keys)
        return self.left.native_keys() | self.right.native_keys()


class RecursiveWeightedMinCutPartitioner(GraphPartitionerBase):
    """Partition a visibility graph by recursive NetworkX weighted min-cuts.

    Args:
        max_cameras: Stop cutting once a part has at most this many cameras.
        overlap_cameras: Boundary cameras copied from each side of a cut into the
            opposite side. Zero keeps the camera sets disjoint.
        max_part_fraction: Largest share of cameras allowed on one side of a cut.
            ``None`` accepts any Stoer–Wagner cut. ``0.8`` rejects cuts above 80/20
            and falls back to a midpoint split.
        overlap_participates_in_cuts: When True, copied cameras join the child group
            and take part in later cuts until every leaf is within ``max_cameras``.
            When False, copies are stamped onto the finished leaves.
    """

    def __init__(
        self,
        max_cameras: int = 40,
        overlap_cameras: int = 5,
        max_part_fraction: float | None = None,
        overlap_participates_in_cuts: bool = False,
    ) -> None:
        super().__init__(process_name="RecursiveWeightedMinCutPartitioner")
        if max_cameras < 1:
            raise ValueError("max_cameras must be >= 1.")
        if overlap_cameras < 0:
            raise ValueError("overlap_cameras must be >= 0.")
        if max_part_fraction is not None and not 0.5 < max_part_fraction <= 1.0:
            raise ValueError("max_part_fraction must be in (0.5, 1].")
        self._max_cameras = max_cameras
        self._overlap_cameras = overlap_cameras
        self._max_part_fraction = max_part_fraction
        self._overlap_participates_in_cuts = overlap_participates_in_cuts

    def __repr__(self) -> str:
        return (
            "RecursiveWeightedMinCutPartitioner("
            f"max_cameras={self._max_cameras}, overlap_cameras={self._overlap_cameras}, "
            f"max_part_fraction={self._max_part_fraction}, "
            f"overlap_participates_in_cuts={self._overlap_participates_in_cuts})"
        )

    def run(
        self,
        graph: VisibilityGraph,
        edge_weights: Mapping[tuple[int, int], int] | None = None,
    ) -> ClusterTree | None:
        """Cluster a visibility graph into a binary tree with boundary overlap.

        Args:
            graph: Visibility edges with ``i < j``.
            edge_weights: Optional positive weight per canonical edge (e.g. inlier
                count). Missing edges, including bridges, use weight 1.
        """
        if len(graph) == 0:
            logger.warning("RecursiveWeightedMinCutPartitioner: no visibility graph provided.")
            return None
        valid_visibility_graph_or_raise(graph)

        edges = sorted({canonical_edge(i, j) for i, j in graph})
        adjacency: dict[int, set[int]] = {}
        weights: dict[int, dict[int, int]] = {}
        weighted_edges = 0
        for i, j in edges:
            weight = 1
            if edge_weights is not None:
                weight = int(edge_weights.get((i, j), edge_weights.get((j, i), 1)))
            weight = max(weight, 1)
            if edge_weights is not None and ((i, j) in edge_weights or (j, i) in edge_weights):
                weighted_edges += 1
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)
            weights.setdefault(i, {})[j] = weight
            weights.setdefault(j, {})[i] = weight

        keys = tuple(sorted(visibility_graph_keys(edges)))
        logger.info(
            "Recursive weighted min-cut: cameras=%d edges=%d weighted=%d/%d "
            "max_cameras=%d overlap_cameras=%d max_part_fraction=%s",
            len(keys),
            len(edges),
            weighted_edges,
            len(edges),
            self._max_cameras,
            self._overlap_cameras,
            self._max_part_fraction,
        )
        if self._overlap_participates_in_cuts:
            self._cut_count = 0
            self._leaf_count = 0
            self._partition_started = time.perf_counter()
            root = self._partition_with_overlap(keys, adjacency, weights)
            logger.info(
                "Recursive weighted min-cut overlap-in-cuts done cuts=%d leaves=%d elapsed=%.1fs",
                self._cut_count,
                self._leaf_count,
                time.perf_counter() - self._partition_started,
            )
        else:
            root = self._bisect(keys, adjacency, weights)
            self._assign_overlap(root, adjacency)
        return self._to_cluster_tree(root, edges)

    def _bisect(
        self,
        keys: tuple[int, ...],
        adjacency: dict[int, set[int]],
        weights: dict[int, dict[int, int]],
    ) -> _PartitionNode:
        """Recursively split ``keys`` until every part is within the size cap."""
        if len(keys) <= self._max_cameras:
            return _PartitionNode(keys=keys)
        parts = self._mincut_parts(keys, weights)
        if parts is None:
            midpoint = max(1, len(keys) // 2)
            parts = (keys[:midpoint], keys[midpoint:])
        left_keys, right_keys = parts
        return _PartitionNode(
            keys=keys,
            left=self._bisect(left_keys, adjacency, weights),
            right=self._bisect(right_keys, adjacency, weights),
        )

    def _partition_with_overlap(
        self,
        keys: tuple[int, ...],
        adjacency: dict[int, set[int]],
        weights: dict[int, dict[int, int]],
        path: str = "C",
    ) -> _PartitionNode:
        """Cut ``keys``, then let copied boundary cameras join the child before later cuts."""
        unique = tuple(sorted(set(keys)))
        if len(unique) <= self._max_cameras:
            self._leaf_count += 1
            logger.info(
                "Recursive weighted min-cut leaf %s cameras=%d leaves=%d cuts=%d elapsed=%.1fs",
                path,
                len(unique),
                self._leaf_count,
                self._cut_count,
                time.perf_counter() - self._partition_started,
            )
            return _PartitionNode(keys=unique)

        cut_started = time.perf_counter()
        parts = self._mincut_parts(unique, weights)
        cut_seconds = time.perf_counter() - cut_started
        used_fallback = parts is None
        if parts is None:
            midpoint = max(1, len(unique) // 2)
            parts = (unique[:midpoint], unique[midpoint:])
        left_exclusive, right_exclusive = parts
        left_members = set(left_exclusive)
        right_members = set(right_exclusive)
        left_keys, right_keys = left_exclusive, right_exclusive
        if self._overlap_cameras > 0:
            left_keys = self._augment(
                left_exclusive, self._boundary_from_side(right_members, left_members, adjacency), len(unique)
            )
            right_keys = self._augment(
                right_exclusive, self._boundary_from_side(left_members, right_members, adjacency), len(unique)
            )
        self._cut_count += 1
        logger.info(
            "Recursive weighted min-cut %s parent=%d left=%d+%d=%d right=%d+%d=%d "
            "shrink=%d/%d cut=%.2fs%s cuts=%d elapsed=%.1fs",
            path,
            len(unique),
            len(left_exclusive),
            len(left_keys) - len(left_exclusive),
            len(left_keys),
            len(right_exclusive),
            len(right_keys) - len(right_exclusive),
            len(right_keys),
            len(unique) - len(left_keys),
            len(unique) - len(right_keys),
            cut_seconds,
            " fallback" if used_fallback else "",
            self._cut_count,
            time.perf_counter() - self._partition_started,
        )
        return _PartitionNode(
            keys=unique,
            left=self._partition_with_overlap(left_keys, adjacency, weights, f"{path}_1"),
            right=self._partition_with_overlap(right_keys, adjacency, weights, f"{path}_2"),
        )

    def _mincut_parts(
        self, keys: tuple[int, ...], weights: dict[int, dict[int, int]]
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Return a 2-way weighted min-cut, or None when the cut is unusable."""
        if len(keys) < 2:
            return None

        graph = nx.Graph()
        graph.add_nodes_from(keys)
        keyset = set(keys)
        for key in keys:
            for neighbor, weight in weights.get(key, {}).items():
                if neighbor in keyset and key < neighbor:
                    graph.add_edge(key, neighbor, weight=int(weight))

        if graph.number_of_edges() == 0:
            return None

        if not nx.is_connected(graph):
            components = sorted((tuple(sorted(component)) for component in nx.connected_components(graph)), key=len)
            # Largest component vs the rest keeps a single bipartition.
            right = components[-1]
            left = tuple(sorted(key for component in components[:-1] for key in component))
            if not left or not right:
                return None
            return self._accept_parts(left, right, len(keys))

        try:
            _cut_value, (part_a, part_b) = nx.stoer_wagner(graph, weight="weight")
        except nx.NetworkXError as exc:
            logger.warning("NetworkX stoer_wagner failed (%s); splitting the key list.", exc)
            return None

        left = tuple(sorted(part_a))
        right = tuple(sorted(part_b))
        if not left or not right:
            return None
        return self._accept_parts(left, right, len(keys))

    def _accept_parts(
        self, left: tuple[int, ...], right: tuple[int, ...], parent_size: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Reject cuts that violate ``max_part_fraction`` when that limit is set."""
        larger = max(len(left), len(right))
        if self._max_part_fraction is not None and larger > self._max_part_fraction * parent_size + 1e-9:
            logger.info(
                "Weighted min-cut rejected: larger side %d/%d exceeds max_part_fraction=%s",
                larger,
                parent_size,
                self._max_part_fraction,
            )
            return None
        return left, right

    def _boundary_from_side(self, source: set[int], destination: set[int], adjacency: dict[int, set[int]]) -> list[int]:
        """Rank ``source`` cameras by how many neighbors they have in ``destination``."""
        scores = {key: 0 for key in source}
        for key in source:
            for neighbor in adjacency.get(key, ()):
                if neighbor in destination:
                    scores[key] += 1
        return self._boundary_cameras(scores)

    @staticmethod
    def _augment(side: tuple[int, ...], added: list[int], parent_size: int) -> tuple[int, ...]:
        """Add cameras from ``added`` while the result stays strictly smaller than ``parent_size``."""
        members = set(side)
        ordered = list(side)
        for key in added:
            if key in members:
                continue
            if len(ordered) + 1 >= parent_size:
                break
            members.add(key)
            ordered.append(key)
        return tuple(sorted(ordered))

    def _assign_overlap(self, node: _PartitionNode, adjacency: dict[int, set[int]]) -> None:
        """Copy boundary cameras into the opposite leaf that shares their edge."""
        if node.left is None or node.right is None or self._overlap_cameras == 0:
            return
        self._assign_overlap(node.left, adjacency)
        self._assign_overlap(node.right, adjacency)

        left_keys = node.left.native_keys()
        right_keys = node.right.native_keys()
        left_score = {key: 0 for key in left_keys}
        right_score = {key: 0 for key in right_keys}
        for key in left_keys:
            for neighbor in adjacency.get(key, ()):
                if neighbor in right_keys:
                    left_score[key] += 1
                    right_score[neighbor] += 1

        for key in self._boundary_cameras(left_score):
            anchors = [neighbor for neighbor in adjacency.get(key, ()) if neighbor in right_keys]
            if not anchors:
                continue
            anchor = min(anchors, key=lambda camera: (-right_score[camera], camera))
            self._find_leaf(node.right, anchor).borrowed.add(key)
        for key in self._boundary_cameras(right_score):
            anchors = [neighbor for neighbor in adjacency.get(key, ()) if neighbor in left_keys]
            if not anchors:
                continue
            anchor = min(anchors, key=lambda camera: (-left_score[camera], camera))
            self._find_leaf(node.left, anchor).borrowed.add(key)

    def _boundary_cameras(self, scores: dict[int, int]) -> list[int]:
        ranked = sorted((key for key, score in scores.items() if score > 0), key=lambda key: (-scores[key], key))
        return ranked[: self._overlap_cameras]

    def _find_leaf(self, node: _PartitionNode, key: int) -> _PartitionNode:
        if node.left is None or node.right is None:
            return node
        if key in node.left.native_keys():
            return self._find_leaf(node.left, key)
        return self._find_leaf(node.right, key)

    def _to_cluster_tree(self, node: _PartitionNode, edges: list[tuple[int, int]]) -> ClusterTree:
        """Store intra-edges and borrowed boundary edges on leaves, and cuts on parents."""
        if node.left is None or node.right is None:
            cameras = set(node.keys) | node.borrowed
            leaf_edges = [
                edge
                for edge in edges
                if edge[0] in cameras and edge[1] in cameras and (edge[0] in node.keys or edge[1] in node.keys)
            ]
            return ClusterTree(value=leaf_edges, children=())

        left_keys = node.left.native_keys()
        right_keys = node.right.native_keys()
        left_only = left_keys - right_keys
        right_only = right_keys - left_keys
        cross_edges = [
            edge
            for edge in edges
            if (edge[0] in left_only and edge[1] in right_only) or (edge[1] in left_only and edge[0] in right_only)
        ]
        return ClusterTree(
            value=cross_edges,
            children=(
                self._to_cluster_tree(node.left, edges),
                self._to_cluster_tree(node.right, edges),
            ),
        )
