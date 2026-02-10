"""
Defining index image pair(s), a visibility graph, and an annotated visibility graph.
"""

from typing import FrozenSet, Iterable, TypeVar

import networkx as nx

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

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


def prune_edges(graph: VisibilityGraph, bad_edges: set[ImageIndexPair]) -> VisibilityGraph:
    """Remove bad edges from a visibility graph.

    Args:
        graph: The original visibility graph.
        bad_edges: Set of edges to remove.

    Returns:
        New visibility graph with bad edges removed.
    """
    return [edge for edge in graph if edge not in bad_edges]


def prune_edges_preserve_connectivity(
    graph: VisibilityGraph,
    bad_edges: set[ImageIndexPair],
) -> tuple[VisibilityGraph, set[ImageIndexPair], set[ImageIndexPair]]:
    """Remove bad edges without disconnecting the graph.

    Bridge edges (whose removal would disconnect the graph) are kept even
    if they appear in ``bad_edges``.

    Args:
        graph: The original visibility graph.
        bad_edges: Set of edges to remove.

    Returns:
        Tuple of (pruned_graph, removed_edges, kept_bridge_edges).
    """
    if not graph or not bad_edges:
        return list(graph), set(), set()

    G = nx.Graph()
    G.add_edges_from(graph)

    # nx.bridges() may yield (u,v) with u > v; canonicalize to (min, max).
    bridge_set: set[ImageIndexPair] = set()
    for u, v in nx.bridges(G):
        bridge_set.add((min(u, v), max(u, v)))

    removable = bad_edges - bridge_set
    kept_as_bridges = bad_edges & bridge_set

    pruned = [edge for edge in graph if edge not in removable]

    if kept_as_bridges:
        logger.info(
            "Connectivity-preserving prune: %d bad edges removed, %d kept as bridges.",
            len(removable),
            len(kept_as_bridges),
        )

    return pruned, removable, kept_as_bridges
