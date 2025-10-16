"""Cluster tree data structures built on top of generic trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Generic, Tuple, TypeVar

from gtsfm.products.visibility_graph import AnnotatedGraph, VisibilityGraph
from gtsfm.utils.tree import Tree

T = TypeVar("T")

ClusterNode = Tree[VisibilityGraph]


def cluster_edges(node: ClusterNode) -> VisibilityGraph:
    """Return the visibility edges stored at this cluster."""
    return node.value


def cluster_local_keys(node: ClusterNode) -> FrozenSet[int]:
    """Keys referenced directly by this cluster's edges."""
    keys: set[int] = set()
    for i, j in node.value:
        keys.add(i)
        keys.add(j)
    return frozenset(keys)


def cluster_all_keys(node: ClusterNode) -> FrozenSet[int]:
    """Return the set of keys contained in this cluster and all descendants."""
    keys = set(cluster_local_keys(node))
    for child in node.children:
        keys.update(cluster_all_keys(child))
    return frozenset(keys)


def cluster_all_edges(node: ClusterNode) -> VisibilityGraph:
    """Return all edges contained in this cluster and its descendants."""
    edges = list(node.value)
    for child in node.children:
        edges.extend(cluster_all_edges(child))
    return edges


def cluster_filter_edges(node: ClusterNode, annotated_graph: AnnotatedGraph[T]) -> AnnotatedGraph[T]:
    """Extract annotated results associated with this cluster's edges."""
    return {edge: annotated_graph[edge] for edge in node.value if edge in annotated_graph}


AnnotatedClusterNode = Tree[AnnotatedGraph[T]]


def annotated_local_keys(node: AnnotatedClusterNode[T]) -> FrozenSet[int]:
    """Keys referenced directly by this cluster's annotations."""
    keys: set[int] = set()
    for i, j in node.value:
        keys.add(i)
        keys.add(j)
    return frozenset(keys)


def annotated_all_keys(node: AnnotatedClusterNode[T]) -> FrozenSet[int]:
    """Return the set of keys contained in this cluster and all descendants."""
    keys = set(annotated_local_keys(node))
    for child in node.children:
        keys.update(annotated_all_keys(child))
    return frozenset(keys)


def annotated_all_annotations(node: AnnotatedClusterNode[T]) -> AnnotatedGraph[T]:
    """Return all annotations contained in this cluster and its descendants."""
    annotations = dict(node.value)
    for child in node.children:
        annotations.update(annotated_all_annotations(child))
    return annotations


@dataclass(frozen=True)
class ClusterTree:
    """Hierarchical cluster tree produced by a graph partitioner."""

    root: ClusterNode | None

    def is_empty(self) -> bool:
        """Return True if the cluster tree has no clusters."""
        return self.root is None

    def leaves(self) -> Tuple[ClusterNode, ...]:
        """Return the leaf clusters."""
        if self.root is None:
            return ()
        return self.root.leaves()

    def __repr__(self) -> str:
        if self.root is None:
            return "ClusterTree(root=None)"

        def _repr(node: ClusterNode, depth: int = 0) -> str:
            indent = "  " * depth
            edges = list(node.value)
            keys = sorted(cluster_local_keys(node))
            s = f"{indent}Cluster(keys={keys}, edges={edges})"
            if node.children:
                for child in node.children:
                    s += "\n" + _repr(child, depth + 1)
            return s

        return f"ClusterTree(\n{_repr(self.root)}\n)"

    def group_by_leaf(self, annotated_graph: AnnotatedGraph[T]) -> list[AnnotatedGraph[T]]:
        """Group annotated results by leaf clusters."""
        return [cluster_filter_edges(leaf, annotated_graph) for leaf in self.leaves()]


@dataclass(frozen=True)
class AnnotatedClusterTree(Generic[T]):
    """Hierarchical cluster tree where edges carry annotations."""

    root: AnnotatedClusterNode[T] | None

    @staticmethod
    def create(cluster_tree: ClusterTree, annotated_graph: AnnotatedGraph[T]) -> "AnnotatedClusterTree[T]":
        """Return an annotated tree given a cluster structure and edge annotations."""

        if cluster_tree.root is None:
            return AnnotatedClusterTree(root=None)

        def _convert(node: ClusterNode) -> AnnotatedClusterNode[T]:
            children = tuple(_convert(child) for child in node.children)
            annotations = cluster_filter_edges(node, annotated_graph)
            return Tree(value=annotations, children=children)

        return AnnotatedClusterTree(root=_convert(cluster_tree.root))

    def is_empty(self) -> bool:
        """Return True if the annotated cluster tree has no clusters."""
        return self.root is None

    def leaves(self) -> Tuple[AnnotatedClusterNode[T], ...]:
        """Return the leaf annotated clusters."""
        if self.root is None:
            return ()
        return self.root.leaves()

    def group_by_leaf(self) -> list[AnnotatedGraph[T]]:
        """Return annotated results grouped by leaf clusters."""
        return [leaf.value for leaf in self.leaves()]
