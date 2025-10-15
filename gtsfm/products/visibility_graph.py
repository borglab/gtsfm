"""
Defining index image pair(s), a visibility graph, and an annotated visibility graph.
"""

from typing import TypeVar

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
