"""Two-way METIS graph bisection shared by recursive partitioners.

``METIS_PartGraphRecursive`` builds an ND-style cluster tree when applied
recursively: each cut is a balanced (optionally weighted) edge-cut. Edge
weights are passed as ``adjwgt`` so weakly matched pairs are preferred cuts.
Optional Kernighan–Lin refinement improves the cut while preserving side sizes.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Mapping
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

_IDX = ctypes.c_int32
_REAL = ctypes.c_float
_METIS_OK = 1
_METIS_NOPTIONS = 40
_METIS_OPTION_SEED = 8
_METIS_OPTION_CONTIG = 10
_METIS_OPTION_NUMBERING = 17

_METIS: ctypes.CDLL | None = None


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


def _metis() -> ctypes.CDLL:
    global _METIS
    if _METIS is None:
        _METIS = _load_metis()
    return _METIS


def refine_bisection(
    left: tuple[int, ...],
    right: tuple[int, ...],
    adjacency: dict[int, set[int]],
    edge_weights: Mapping[tuple[int, int], int] | None = None,
    max_iter: int = 10,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Improve a 2-way cut with Kernighan–Lin while preserving side sizes.

    Args:
        left: Cameras on the first side of the cut.
        right: Cameras on the second side of the cut.
        adjacency: Undirected neighbor sets.
        edge_weights: Optional positive weight per canonical edge.
        max_iter: Maximum Kernighan–Lin pass count.

    Returns:
        Refined ``(left, right)`` key tuples. Returns the input partition unchanged
        when the induced subgraph has no edges or refinement fails.
    """
    if not left or not right:
        return left, right
    keys = left + right
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
    if graph.number_of_edges() == 0:
        return left, right
    try:
        part_a, part_b = kernighan_lin_bisection(
            graph, partition=(set(left), set(right)), max_iter=max_iter, weight="weight", seed=0
        )
    except Exception as exc:
        logger.warning("Kernighan–Lin refinement failed (%s); keeping METIS cut.", exc)
        return left, right
    refined_left = tuple(sorted(part_a))
    refined_right = tuple(sorted(part_b))
    if not refined_left or not refined_right:
        return left, right
    return refined_left, refined_right


def bisect_graph(
    keys: tuple[int, ...],
    adjacency: dict[int, set[int]],
    max_part_fraction: float | None = None,
    edge_weights: Mapping[tuple[int, int], int] | None = None,
    refine_kl: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return a balanced 2-way METIS partition, or None when METIS cannot cut.

    Args:
        keys: Cameras to partition (order is preserved in the CSR layout).
        adjacency: Undirected neighbor sets.
        max_part_fraction: Largest share of cameras allowed on one side.
            ``None`` or ``0.5`` keeps METIS's default ~50/50 balance. ``0.8``
            allows about 80/20.
        edge_weights: Optional positive weight per canonical edge (e.g. inlier
            count). When set, METIS minimizes the weighted edge-cut via ``adjwgt``.
        refine_kl: When True, run Kernighan–Lin on the METIS cut (side sizes kept).

    Returns:
        ``(left, right)`` key tuples, or ``None`` if the graph has no edges or METIS fails.
    """
    if len(keys) < 2:
        return None

    local_index = {key: index for index, key in enumerate(keys)}
    neighbor_lists = [
        sorted(local_index[neighbor] for neighbor in adjacency.get(key, ()) if neighbor in local_index) for key in keys
    ]
    if not any(neighbor_lists):
        return None

    xadj_values = [0]
    adjncy_values: list[int] = []
    adjwgt_values: list[int] = []
    for key, neighbors in zip(keys, neighbor_lists):
        for neighbor_idx in neighbors:
            neighbor = keys[neighbor_idx]
            adjncy_values.append(neighbor_idx)
            weight = 1
            if edge_weights is not None:
                weight = int(edge_weights.get((key, neighbor), edge_weights.get((neighbor, key), 1)))
            adjwgt_values.append(max(weight, 1))
        xadj_values.append(len(adjncy_values))

    nvtxs = _IDX(len(keys))
    ncon = _IDX(1)
    nparts = _IDX(2)
    xadj = (_IDX * len(xadj_values))(*xadj_values)
    adjncy = (_IDX * len(adjncy_values))(*adjncy_values)
    adjwgt = None
    if edge_weights is not None:
        adjwgt = (_IDX * len(adjwgt_values))(*adjwgt_values)
    options = (_IDX * _METIS_NOPTIONS)()
    if _metis().METIS_SetDefaultOptions(options) != _METIS_OK:
        return None
    options[_METIS_OPTION_SEED] = 0
    options[_METIS_OPTION_CONTIG] = 1
    options[_METIS_OPTION_NUMBERING] = 0
    edge_cut = _IDX()
    part = (_IDX * len(keys))()
    # Equal targets are 1/2. ubvec 1.6 means a side may hold 0.5 * 1.6 = 80%.
    # ubvec=1.0 (fraction 0.5) is too tight for odd counts; use METIS default.
    ubvec = None
    if max_part_fraction is not None and max_part_fraction > 0.5 + 1e-12:
        ubvec = (_REAL * 1)(2.0 * max_part_fraction)
    status = _metis().METIS_PartGraphRecursive(
        ctypes.byref(nvtxs),
        ctypes.byref(ncon),
        xadj,
        adjncy,
        None,
        None,
        adjwgt,
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
    if refine_kl:
        left, right = refine_bisection(left, right, adjacency, edge_weights=edge_weights)
    return left, right
