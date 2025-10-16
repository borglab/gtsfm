"""
Defining index image pair(s), a visibility graph, and an annotated visibility graph.
"""

from typing import FrozenSet, Iterable, TypeVar

ImageIndexPair = tuple[int, int]
ImageIndexPairs = list[ImageIndexPair]  # list of (i,j) index pairs

VisibilityGraph = ImageIndexPairs  # if we mean the graph and not a subset of edges


def valid_visibility_graph_or_raise(graph: VisibilityGraph) -> None:
    """Check if the given visibility graph is valid, raise if not."""
    for i, j in graph:
        if i == j:
            raise ValueError(f"VisibilityGraph contains self-loop ({i}, {j}).")
        if i > j:
            raise ValueError(f"VisibilityGraph contains invalid pair ({i}, {j}): i must be less than j.")


T = TypeVar("T")

AnnotatedGraph = dict[ImageIndexPair, T]  # (i,j) -> T


def visibility_graph_keys(edges: Iterable[ImageIndexPair]) -> FrozenSet[int]:
    """Return the set of vertex indices referenced by `edges`."""
    keys: set[int] = set()
    for i, j in edges:
        keys.add(i)
        keys.add(j)
    return frozenset(keys)


def filter_annotations_by_edges(edges: Iterable[ImageIndexPair], annotations: AnnotatedGraph[T]) -> AnnotatedGraph[T]:
    """Restrict `annotations` to entries keyed by `edges`."""
    return {edge: annotations[edge] for edge in edges if edge in annotations}
