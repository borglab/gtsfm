"""Data structures for hierarchical clustering of visibility graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Tuple, TypeVar

from gtsfm.products.visibility_graph import AnnotatedGraph, VisibilityGraph

T = TypeVar("T")


@dataclass(frozen=True)
class Cluster:
    """Node in a hierarchical clustering tree."""

    keys: FrozenSet[int]
    edges: VisibilityGraph
    children: Tuple["Cluster", ...] = ()

    def is_leaf(self) -> bool:
        """Return True if the cluster is a leaf (has no children)."""
        return len(self.children) == 0

    def all_keys(self) -> FrozenSet[int]:
        """Return the set of keys contained in this cluster and all descendants."""
        if self.is_leaf():
            return self.keys
        descendant_keys = set(self.keys)
        for child in self.children:
            descendant_keys.update(child.all_keys())
        return frozenset(descendant_keys)

    def all_edges(self) -> VisibilityGraph:
        """Return all edges contained in this cluster and its descendants."""
        edges = list(self.edges)
        for child in self.children:
            edges.extend(child.all_edges())
        return edges

    def extract(self, annotated_graph: AnnotatedGraph[T]) -> AnnotatedGraph[T]:
        """Extract annotated results associated with this cluster's edges."""
        return {edge: annotated_graph[edge] for edge in self.edges if edge in annotated_graph}


@dataclass(frozen=True)
class Clustering:
    """Hierarchical clustering produced by a graph partitioner."""

    root: Cluster | None

    def is_empty(self) -> bool:
        """Return True if the clustering has no clusters."""
        return self.root is None

    def leaves(self) -> Tuple[Cluster, ...]:
        """Return the leaf clusters."""
        if self.root is None:
            return ()

        leaves: List[Cluster] = []

        def _collect(cluster: Cluster) -> None:
            if cluster.is_leaf():
                leaves.append(cluster)
                return
            for child in cluster.children:
                _collect(child)

        _collect(self.root)
        return tuple(leaves)

    def group_by_leaf(self, annotated_graph: AnnotatedGraph[T]) -> List[AnnotatedGraph[T]]:
        """Group annotated results by leaf clusters."""
        return [leaf.extract(annotated_graph) for leaf in self.leaves()]
