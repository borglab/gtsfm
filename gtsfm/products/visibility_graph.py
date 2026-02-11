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
    bad_edges: list[ImageIndexPair],
) -> tuple[VisibilityGraph, set[ImageIndexPair], set[ImageIndexPair]]:
    """Remove bad edges without disconnecting the graph.

    Edges are removed iteratively in the order given by ``bad_edges``
    (caller should sort worst-first: zero-track edges before high-reproj-error).
    After tentatively removing each edge, a connectivity check is performed;
    if the endpoints become disconnected the edge is kept.

    Args:
        graph: The original visibility graph.
        bad_edges: Ordered list of edges to try removing (worst first).

    Returns:
        Tuple of (pruned_graph, removed_edges, kept_edges).
    """
    if not graph or not bad_edges:
        return list(graph), set(), set()

    G = nx.Graph()
    G.add_edges_from(graph)

    bad_edge_set = set(bad_edges)
    removed: set[ImageIndexPair] = set()
    kept: set[ImageIndexPair] = set()

    for u, v in bad_edges:
        if not G.has_edge(u, v):
            # Edge not in graph (duplicate in bad list or not in original graph).
            continue
        G.remove_edge(u, v)
        if nx.has_path(G, u, v):
            removed.add((u, v))
        else:
            # Removing this edge would disconnect the graph — put it back.
            G.add_edge(u, v)
            kept.add((u, v))

    pruned = [edge for edge in graph if edge not in removed]

    if kept:
        logger.info(
            "Connectivity-preserving prune: %d bad edges removed, %d kept for connectivity.",
            len(removed),
            len(kept),
        )

    return pruned, removed, kept
