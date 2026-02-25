"""Utility helpers shared across graph partitioners."""

from __future__ import annotations

from gtsfm.products.visibility_graph import VisibilityGraph


def min_key(keys: set[int]) -> int:
    """Return smallest key or a large sentinel for empty sets."""
    return min(keys) if keys else 10**18


def canonical_edge(i: int, j: int) -> tuple[int, int]:
    """Return edge in canonical undirected order."""
    return (i, j) if i < j else (j, i)


def count_cross_edges(keys_a: set[int], keys_b: set[int], graph: VisibilityGraph) -> int:
    """Count edges crossing between two key sets."""
    return sum(1 for i, j in graph if (i in keys_a and j in keys_b) or (i in keys_b and j in keys_a))


def partition_local_keys(keys: list[int], num_bins: int) -> list[set[int]]:
    """Deterministically split sorted keys into balanced contiguous bins."""
    n = len(keys)
    base = n // num_bins
    rem = n % num_bins
    bins: list[set[int]] = []
    start = 0
    for idx in range(num_bins):
        size = base + (1 if idx < rem else 0)
        bins.append(set(keys[start:start + size]))
        start += size
    return bins


def graph_adjacency(graph: VisibilityGraph) -> dict[int, set[int]]:
    """Build undirected adjacency list from visibility graph."""
    adjacency: dict[int, set[int]] = {}
    for i, j in graph:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    return adjacency


def build_edges_for_keyset(keys: set[int], graph: VisibilityGraph) -> list[tuple[int, int]]:
    """Build a compact edge set to realize a target set of local keys."""
    if len(keys) < 2:
        return []
    sorted_keys = sorted(keys)
    graph_edges = {canonical_edge(i, j) for i, j in graph}
    edges: list[tuple[int, int]] = []
    for idx in range(len(sorted_keys) - 1):
        a, b = sorted_keys[idx], sorted_keys[idx + 1]
        edge = canonical_edge(a, b)
        edges.append(edge if edge in graph_edges else edge)
    return sorted(set(edges))
