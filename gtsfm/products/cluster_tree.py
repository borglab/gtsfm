"""Cluster tree data structures built on top of generic trees."""

from __future__ import annotations

from typing import FrozenSet, Generic, Optional, TypeVar, cast

from gtsfm.products.visibility_graph import (
    AnnotatedGraph,
    VisibilityGraph,
    filter_annotations_by_edges,
    visibility_graph_keys,
)
from gtsfm.utils.tree import Tree

T = TypeVar("T")


class ClusterTree(Tree[VisibilityGraph]):
    """Tree node representing a hierarchical cluster of visibility edges."""

    def _child_clusters(self) -> tuple["ClusterTree", ...]:
        return cast(tuple["ClusterTree", ...], self.children)

    def local_keys(self) -> FrozenSet[int]:
        """Keys referenced directly by this cluster's edges."""
        return visibility_graph_keys(self.value)

    def all_keys(self) -> FrozenSet[int]:
        """Return the set of unique keys contained in this cluster and all descendants."""
        keys = set(self.local_keys())
        for child in self._child_clusters():
            keys.update(child.all_keys())
        return frozenset(keys)

    def all_edges(self) -> FrozenSet[tuple[int, int]]:
        """Return all *unique* edges contained in this cluster and its descendants."""
        edges = set(self.value)
        for child in self._child_clusters():
            edges.update(child.all_edges())
        return frozenset(edges)

    def filter_annotations(self, annotated_graph: AnnotatedGraph[T]) -> AnnotatedGraph[T]:
        """Extract annotated results associated with this cluster's edges."""
        return filter_annotations_by_edges(self.value, annotated_graph)

    def group_by_leaf(self, annotated_graph: AnnotatedGraph[T]) -> list[AnnotatedGraph[T]]:
        """Group annotated results by leaf clusters."""
        return [leaf.filter_annotations(annotated_graph) for leaf in self.leaves()]

    def to_annotated(self, annotated_graph: AnnotatedGraph[T]) -> "AnnotatedClusterTree[T]":
        """Convert this cluster tree into an annotated cluster tree."""
        children = tuple(child.to_annotated(annotated_graph) for child in self._child_clusters())
        annotations = filter_annotations_by_edges(self.value, annotated_graph)
        return AnnotatedClusterTree(value=annotations, children=children)

    def __repr__(self) -> str:
        """Return an indented string representation of the cluster tree, showing local keys at each node."""

        def repr_helper(node: "ClusterTree", indent: int = 0) -> str:
            prefix = " " * indent
            node_type = "Leaf" if node.is_leaf() else "Cluster"
            # Sorting keys makes the output deterministic and easier to read
            keys = sorted(list(node.local_keys()))
            lines = [f"{prefix}{node_type} ({len(keys)}): {sorted(keys)}"]
            for child in node._child_clusters():
                lines.append(repr_helper(child, indent + 2))
            return "\n".join(lines)

        return repr_helper(self)


class AnnotatedClusterTree(Tree[AnnotatedGraph[T]], Generic[T]):
    """Cluster tree where edges carry annotations."""

    def _child_clusters(self) -> tuple["AnnotatedClusterTree[T]", ...]:
        return cast(tuple["AnnotatedClusterTree[T]", ...], self.children)

    def local_keys(self) -> FrozenSet[int]:
        """Keys referenced directly by this cluster's annotations."""
        return visibility_graph_keys(self.value.keys())

    def all_keys(self) -> FrozenSet[int]:
        """Return the set of keys contained in this cluster and all descendants."""

        def reducer(annotations: AnnotatedGraph[T], child_keys: tuple[FrozenSet[int], ...]) -> FrozenSet[int]:
            merged: set[int] = set()
            for key_set in child_keys:
                merged.update(key_set)
            merged.update(visibility_graph_keys(annotations.keys()))
            return frozenset(merged)

        return self.fold(reducer)

    def all_annotations(self) -> AnnotatedGraph[T]:
        """Return all annotations contained in this cluster and its descendants."""

        def reducer(
            annotations: AnnotatedGraph[T], child_annotations: tuple[AnnotatedGraph[T], ...]
        ) -> AnnotatedGraph[T]:
            merged: AnnotatedGraph[T] = dict(annotations)
            for child in child_annotations:
                merged.update(child)
            return merged

        return self.fold(reducer)

    def group_by_leaf(self) -> list[AnnotatedGraph[T]]:
        """Return annotated results grouped by leaf clusters."""
        return [leaf.value for leaf in self.leaves()]

    @classmethod
    def from_cluster_tree(
        cls, cluster_tree: Optional[ClusterTree], annotated_graph: AnnotatedGraph[T]
    ) -> Optional["AnnotatedClusterTree[T]"]:
        """Return an annotated tree given a cluster structure and edge annotations."""
        if cluster_tree is None:
            return None
        return cluster_tree.to_annotated(annotated_graph)
