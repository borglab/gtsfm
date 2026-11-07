"""NetworkX bipartition helpers for ND-style recursive cluster trees.

Uses only NetworkX (plus its SciPy backend where NetworkX calls into it):

* ``spectral`` / ``nested_dissection``: Fiedler ordering + normalized-cut sweep
  (algebraic / spectral nested dissection style separator).
* ``stoer_wagner``: exact weighted min-cut, rejected when unbalanced.
* ``kernighan_lin``: local refine that preserves side sizes.

``auto`` tries spectral Ncut, then balanced Stoer–Wagner, then midpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
from networkx.algorithms.cuts import normalized_cut_size

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

NetworkXCutter = Literal["auto", "spectral", "nested_dissection", "stoer_wagner"]


def _balance_limit(parent_size: int, max_part_fraction: float | None) -> int:
    """Largest side allowed (matches recursive METIS / WGM semantics)."""
    if max_part_fraction is None or max_part_fraction <= 0.5 + 1e-12:
        return (parent_size + 1) // 2
    return math.floor(max_part_fraction * parent_size + 1e-9)


def _build_graph(
    keys: tuple[int, ...],
    adjacency: dict[int, set[int]],
    edge_weights: Mapping[tuple[int, int], int] | None,
) -> nx.Graph:
    """Build an undirected NetworkX graph on ``keys`` with optional edge weights."""
    graph = nx.Graph()
    graph.add_nodes_from(keys)
    keyset = set(keys)
    for key in keys:
        for neighbor in adjacency.get(key, ()):
            if neighbor not in keyset or key >= neighbor:
                continue
            weight = 1
            if edge_weights is not None:
                weight = int(edge_weights.get((key, neighbor), edge_weights.get((neighbor, key), 1)))
            graph.add_edge(key, neighbor, weight=max(weight, 1))
    return graph


def refine_bisection_kl(
    left: tuple[int, ...],
    right: tuple[int, ...],
    adjacency: dict[int, set[int]],
    edge_weights: Mapping[tuple[int, int], int] | None = None,
    max_iter: int = 10,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Kernighan–Lin refine that preserves side sizes."""
    if not left or not right:
        return left, right
    graph = _build_graph(left + right, adjacency, edge_weights)
    if graph.number_of_edges() == 0:
        return left, right
    try:
        part_a, part_b = kernighan_lin_bisection(
            graph, partition=(set(left), set(right)), max_iter=max_iter, weight="weight", seed=0
        )
    except Exception as exc:
        logger.warning("NetworkX Kernighan–Lin failed (%s); keeping prior cut.", exc)
        return left, right
    refined_left = tuple(sorted(part_a))
    refined_right = tuple(sorted(part_b))
    if not refined_left or not refined_right:
        return left, right
    return refined_left, refined_right


def bisect_stoer_wagner(
    keys: tuple[int, ...],
    adjacency: dict[int, set[int]],
    max_part_fraction: float | None = None,
    edge_weights: Mapping[tuple[int, int], int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Exact weighted min-cut, or None when unbalanced / unusable."""
    if len(keys) < 2:
        return None
    graph = _build_graph(keys, adjacency, edge_weights)
    if graph.number_of_edges() == 0:
        return None
    if not nx.is_connected(graph):
        components = sorted((tuple(sorted(c)) for c in nx.connected_components(graph)), key=len)
        right = components[-1]
        left = tuple(sorted(key for component in components[:-1] for key in component))
        if not left or not right:
            return None
        return _accept_balanced(left, right, len(keys), max_part_fraction)
    try:
        _cut_value, (part_a, part_b) = nx.stoer_wagner(graph, weight="weight")
    except nx.NetworkXError as exc:
        logger.warning("NetworkX stoer_wagner failed (%s).", exc)
        return None
    left = tuple(sorted(part_a))
    right = tuple(sorted(part_b))
    if not left or not right:
        return None
    return _accept_balanced(left, right, len(keys), max_part_fraction)


def bisect_spectral_ncut(
    keys: tuple[int, ...],
    adjacency: dict[int, set[int]],
    max_part_fraction: float | None = None,
    edge_weights: Mapping[tuple[int, int], int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Balanced bipartition via Fiedler order + NetworkX normalized-cut sweep.

    This is the spectral / algebraic nested-dissection style separator: order by
    the Fiedler vector, then pick the threshold that minimizes ``normalized_cut_size``
    subject to the balance limit.
    """
    if len(keys) < 2:
        return None
    graph = _build_graph(keys, adjacency, edge_weights)
    if graph.number_of_edges() == 0:
        return None
    n = len(keys)
    try:
        # spectral_ordering contiguous-izes components and sorts each by Fiedler.
        order = list(nx.spectral_ordering(graph, weight="weight", normalized=True, seed=0))
    except Exception as exc:
        logger.warning("NetworkX spectral_ordering failed (%s).", exc)
        try:
            part_a, part_b = nx.spectral_bisection(graph, weight="weight", normalized=True, seed=0)
            left = tuple(sorted(part_a))
            right = tuple(sorted(part_b))
            return _accept_balanced(left, right, n, max_part_fraction)
        except Exception as exc2:
            logger.warning("NetworkX spectral_bisection failed (%s).", exc2)
            return None

    if len(order) != n:
        return None
    limit = _balance_limit(n, max_part_fraction)
    lo = max(1, n - limit)
    hi = min(n - 1, limit)
    if lo > hi:
        return None

    best_ncut = float("inf")
    best_k: int | None = None
    mid = n / 2.0
    for k in range(lo, hi + 1):
        left_set = set(order[:k])
        right_set = set(order[k:])
        try:
            ncut = float(normalized_cut_size(graph, left_set, right_set, weight="weight"))
        except Exception:
            continue
        better = ncut < best_ncut - 1e-15
        tie = best_k is not None and abs(ncut - best_ncut) <= 1e-15 and abs(k - mid) < abs(best_k - mid)
        if best_k is None or better or tie:
            best_ncut = ncut
            best_k = k
    if best_k is None:
        return None
    left = tuple(sorted(order[:best_k]))
    right = tuple(sorted(order[best_k:]))
    return left, right


def _accept_balanced(
    left: tuple[int, ...],
    right: tuple[int, ...],
    parent_size: int,
    max_part_fraction: float | None,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Reject cuts that violate the balance limit."""
    larger = max(len(left), len(right))
    if larger > _balance_limit(parent_size, max_part_fraction):
        return None
    return left, right


def bisect_graph(
    keys: tuple[int, ...],
    adjacency: dict[int, set[int]],
    max_part_fraction: float | None = None,
    edge_weights: Mapping[tuple[int, int], int] | None = None,
    cutter: NetworkXCutter = "auto",
    refine_kl: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return a balanced NetworkX bipartition, or None when every cutter fails.

    Args:
        keys: Cameras to partition.
        adjacency: Undirected neighbor sets.
        max_part_fraction: Largest share allowed on one side (``None`` ≈ 50/50).
        edge_weights: Optional positive weight per canonical edge.
        cutter: ``auto`` | ``spectral`` | ``nested_dissection`` | ``stoer_wagner``.
            ``nested_dissection`` and ``spectral`` both use the Fiedler+Ncut sweep.
        refine_kl: Run Kernighan–Lin after an accepted cut.

    Returns:
        ``(left, right)`` or ``None``.
    """
    if len(keys) < 2:
        return None

    parts: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    if cutter in ("auto", "spectral", "nested_dissection"):
        parts = bisect_spectral_ncut(keys, adjacency, max_part_fraction, edge_weights)
    if parts is None and cutter in ("auto", "stoer_wagner"):
        parts = bisect_stoer_wagner(keys, adjacency, max_part_fraction, edge_weights)
    if parts is None:
        return None
    if refine_kl:
        parts = refine_bisection_kl(parts[0], parts[1], adjacency, edge_weights)
        # KL preserves sizes, so balance still holds.
    return parts
