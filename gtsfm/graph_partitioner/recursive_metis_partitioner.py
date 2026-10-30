"""Binary cluster tree from recursive METIS bisection.

Each cut is a two-way ``METIS_PartGraphRecursive`` partition of the cameras in
that node. The recursion stops at ``max_cameras``. Boundary cameras are then
copied across each cut so both reconstructions estimate them. By default the
copy lands on the finished leaf. ``overlap_participates_in_cuts`` instead adds
those cameras to the child group before later cuts, and those cuts continue
until every leaf is within ``max_cameras``.
"""

from __future__ import annotations

import ctypes
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.graph_partitioner.partition_utils import canonical_edge
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.visibility_graph import VisibilityGraph, valid_visibility_graph_or_raise, visibility_graph_keys

logger = logger_utils.get_logger()

_IDX = ctypes.c_int32
_REAL = ctypes.c_float
_METIS_OK = 1
_METIS_NOPTIONS = 40
_METIS_OPTION_SEED = 8
_METIS_OPTION_CONTIG = 10
_METIS_OPTION_NUMBERING = 17


def _load_metis() -> ctypes.CDLL:
    """Load the METIS shared library shipped with the GTSFM environment."""
    candidates = []
    env_path = os.environ.get("METIS_LIB")
    if env_path:
        candidates.append(Path(env_path))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / "libmetis.so")
    candidates.append(Path("/usr/lib/x86_64-linux-gnu/libmetis.so"))
    for path in candidates:
        if path.is_file():
            library = ctypes.CDLL(str(path))
            library.METIS_SetDefaultOptions.argtypes = [ctypes.POINTER(_IDX)]
            library.METIS_SetDefaultOptions.restype = ctypes.c_int
            library.METIS_PartGraphRecursive.argtypes = [
                ctypes.POINTER(_IDX),
                ctypes.POINTER(_IDX),
                ctypes.POINTER(_IDX),
                ctypes.POINTER(_IDX),
                ctypes.POINTER(_IDX),
                ctypes.POINTER(_IDX),
                ctypes.POINTER(_IDX),
                ctypes.POINTER(_IDX),
                ctypes.POINTER(_REAL),
                ctypes.POINTER(_REAL),
                ctypes.POINTER(_IDX),
                ctypes.POINTER(_IDX),
                ctypes.POINTER(_IDX),
            ]
            library.METIS_PartGraphRecursive.restype = ctypes.c_int
            return library
    raise RuntimeError("METIS_PartGraphRecursive requires libmetis.so. Set METIS_LIB to its path.")


_METIS: ctypes.CDLL | None = None


def _metis() -> ctypes.CDLL:
    global _METIS
    if _METIS is None:
        _METIS = _load_metis()
    return _METIS


@dataclass
class _PartitionNode:
    """Disjoint camera set produced by one recursive bisection."""

    keys: tuple[int, ...]
    left: "_PartitionNode | None" = None
    right: "_PartitionNode | None" = None
    borrowed: set[int] = field(default_factory=set)

    def native_keys(self) -> set[int]:
        if self.left is None or self.right is None:
            return set(self.keys)
        return self.left.native_keys() | self.right.native_keys()


class RecursiveMetisPartitioner(GraphPartitionerBase):
    """Partition a visibility graph by recursive two-way METIS cuts.

    Args:
        max_cameras: Stop cutting once a part has at most this many cameras.
        overlap_cameras: Boundary cameras copied from each side of a cut into the
            opposite side. Zero keeps the camera sets disjoint.
        max_part_fraction: Largest share of the cameras allowed on one side of a cut.
            ``None`` keeps METIS's default, about a 50/50 split. ``0.8`` allows 80/20.
        overlap_participates_in_cuts: When True, copied cameras join the child group
            and take part in later cuts. Cutting continues until every leaf is within
            ``max_cameras``. When False, copies are stamped onto the finished leaves.
    """

    def __init__(
        self,
        max_cameras: int = 40,
        overlap_cameras: int = 5,
        max_part_fraction: float | None = None,
        overlap_participates_in_cuts: bool = False,
    ) -> None:
        super().__init__(process_name="RecursiveMetisPartitioner")
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
            "RecursiveMetisPartitioner("
            f"max_cameras={self._max_cameras}, overlap_cameras={self._overlap_cameras}, "
            f"max_part_fraction={self._max_part_fraction}, "
            f"overlap_participates_in_cuts={self._overlap_participates_in_cuts})"
        )

    def run(
        self,
        graph: VisibilityGraph,
        edge_weights: Mapping[tuple[int, int], int] | None = None,
    ) -> ClusterTree | None:
        """Cluster a visibility graph into a binary tree with boundary overlap."""
        del edge_weights  # unused; METIS cuts here are unweighted
        if len(graph) == 0:
            logger.warning("RecursiveMetisPartitioner: no visibility graph provided.")
            return None
        valid_visibility_graph_or_raise(graph)

        edges = sorted({canonical_edge(i, j) for i, j in graph})
        adjacency: dict[int, set[int]] = {}
        for i, j in edges:
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)
        keys = tuple(sorted(visibility_graph_keys(edges)))
        if self._overlap_participates_in_cuts:
            self._cut_count = 0
            self._leaf_count = 0
            self._partition_started = time.perf_counter()
            logger.info(
                "Recursive METIS overlap-in-cuts start cameras=%d max_cameras=%d "
                "overlap_cameras=%d max_part_fraction=%s",
                len(keys),
                self._max_cameras,
                self._overlap_cameras,
                self._max_part_fraction,
            )
            root = self._partition_with_overlap(keys, adjacency)
            logger.info(
                "Recursive METIS overlap-in-cuts done cuts=%d leaves=%d elapsed=%.1fs",
                self._cut_count,
                self._leaf_count,
                time.perf_counter() - self._partition_started,
            )
        else:
            root = self._bisect(keys, adjacency)
            self._assign_overlap(root, adjacency)
        return self._to_cluster_tree(root, edges)

    def _bisect(self, keys: tuple[int, ...], adjacency: dict[int, set[int]]) -> _PartitionNode:
        """Recursively split ``keys`` until every part is within the size cap."""
        if len(keys) <= self._max_cameras:
            return _PartitionNode(keys=keys)
        parts = self._metis_parts(keys, adjacency)
        if parts is None:
            midpoint = max(1, len(keys) // 2)
            parts = (keys[:midpoint], keys[midpoint:])
        left_keys, right_keys = parts
        return _PartitionNode(
            keys=keys,
            left=self._bisect(left_keys, adjacency),
            right=self._bisect(right_keys, adjacency),
        )

    def _partition_with_overlap(
        self, keys: tuple[int, ...], adjacency: dict[int, set[int]], path: str = "C"
    ) -> _PartitionNode:
        """Cut ``keys``, then let copied boundary cameras join the child before later cuts.

        Each recursive call receives a strict subset of its parent's cameras: a child of
        a cut is smaller than the cut, and copies from the sibling are added only while
        the child stays smaller than the parent. A call that arrives with at most
        ``max_cameras`` is a leaf, so every leaf is within the cap.
        """
        unique = tuple(sorted(set(keys)))
        if len(unique) <= self._max_cameras:
            self._leaf_count += 1
            logger.info(
                "Recursive METIS leaf %s cameras=%d leaves=%d cuts=%d elapsed=%.1fs",
                path,
                len(unique),
                self._leaf_count,
                self._cut_count,
                time.perf_counter() - self._partition_started,
            )
            return _PartitionNode(keys=unique)
        metis_started = time.perf_counter()
        parts = self._metis_parts(unique, adjacency)
        metis_seconds = time.perf_counter() - metis_started
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
            "Recursive METIS cut %s parent=%d left=%d+%d=%d right=%d+%d=%d "
            "shrink=%d/%d metis=%.2fs%s cuts=%d elapsed=%.1fs",
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
            metis_seconds,
            " fallback" if used_fallback else "",
            self._cut_count,
            time.perf_counter() - self._partition_started,
        )
        return _PartitionNode(
            keys=unique,
            left=self._partition_with_overlap(left_keys, adjacency, f"{path}_1"),
            right=self._partition_with_overlap(right_keys, adjacency, f"{path}_2"),
        )

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

    def _metis_parts(
        self, keys: tuple[int, ...], adjacency: dict[int, set[int]]
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Return a balanced 2-way partition, or None when METIS cannot cut."""
        local_index = {key: index for index, key in enumerate(keys)}
        neighbor_lists = [
            sorted(local_index[neighbor] for neighbor in adjacency.get(key, ()) if neighbor in local_index)
            for key in keys
        ]
        if not any(neighbor_lists):
            return None

        xadj_values = [0]
        adjncy_values: list[int] = []
        for neighbors in neighbor_lists:
            adjncy_values.extend(neighbors)
            xadj_values.append(len(adjncy_values))

        nvtxs = _IDX(len(keys))
        ncon = _IDX(1)
        nparts = _IDX(2)
        xadj = (_IDX * len(xadj_values))(*xadj_values)
        adjncy = (_IDX * len(adjncy_values))(*adjncy_values)
        options = (_IDX * _METIS_NOPTIONS)()
        if _metis().METIS_SetDefaultOptions(options) != _METIS_OK:
            return None
        options[_METIS_OPTION_SEED] = 0
        options[_METIS_OPTION_CONTIG] = 1
        options[_METIS_OPTION_NUMBERING] = 0
        edge_cut = _IDX()
        part = (_IDX * len(keys))()
        # Equal targets are 1/2. ubvec 1.6 means a side may hold 0.5 * 1.6 = 80%.
        ubvec = None
        if self._max_part_fraction is not None:
            ubvec = (_REAL * 1)(2.0 * self._max_part_fraction)
        status = _metis().METIS_PartGraphRecursive(
            ctypes.byref(nvtxs),
            ctypes.byref(ncon),
            xadj,
            adjncy,
            None,
            None,
            None,
            ctypes.byref(nparts),
            None,
            ubvec,
            options,
            ctypes.byref(edge_cut),
            part,
        )
        if status != _METIS_OK:
            logger.warning("METIS_PartGraphRecursive failed with status %s; splitting the key list.", status)
            return None
        left = tuple(key for key, label in zip(keys, part) if int(label) == 0)
        right = tuple(key for key, label in zip(keys, part) if int(label) == 1)
        if not left or not right:
            return None
        return left, right

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
        # A camera copied into both sides stays with each subtree. Only an edge between
        # the exclusive parts is a cut edge.
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
