"""Bridge reconnection for disconnected viewgraph components.

Given a visibility graph and the underlying similarity matrix, identifies
disconnected components and adds bridge edges to reconnect island components
to the main (largest) component.

Authors: Kathir Gounder
"""

from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import torch

import gtsfm.utils.logger as logger_utils
from gtsfm.products.visibility_graph import VisibilityGraph

logger = logger_utils.get_logger()


@dataclass(frozen=True)
class BridgeReconnectionResult:
    """Result of the bridge reconnection process."""

    original_graph: VisibilityGraph
    reconnected_graph: VisibilityGraph
    bridge_edges: List[Tuple[int, int]]
    bridge_similarities: List[float]
    num_components_before: int
    num_components_after: int
    components_reconnected: int
    components_unreachable: int  # islands with no bridge >= min_bridge_sim


def reconnect_visibility_graph(
    visibility_graph: VisibilityGraph,
    similarity_matrix: torch.Tensor,
    min_bridge_similarity: float = 0.25,
    top_k_per_component: int = 10,
    min_component_size: int = 3,
) -> BridgeReconnectionResult:
    """Add bridge edges to reconnect disconnected island components to the main component.

    For each non-main connected component with at least ``min_component_size`` cameras,
    finds the top-K cross-component edges to the main component with similarity >=
    ``min_bridge_similarity`` and adds them to the visibility graph.

    This replicates the pipeline's edge selection logic: the similarity matrix uses
    the upper triangle convention (i < j), and all returned edges satisfy i < j.

    Args:
        visibility_graph: Original list of (i, j) pairs with i < j.
        similarity_matrix: (N, N) similarity matrix (CPU tensor).
        min_bridge_similarity: Minimum similarity for a bridge edge to be added.
        top_k_per_component: Maximum number of bridge edges to add per island component.
        min_component_size: Minimum number of cameras in an island to attempt reconnection.

    Returns:
        BridgeReconnectionResult with the reconnected graph and diagnostics.
    """
    if len(visibility_graph) == 0:
        return BridgeReconnectionResult(
            original_graph=visibility_graph,
            reconnected_graph=visibility_graph,
            bridge_edges=[],
            bridge_similarities=[],
            num_components_before=0,
            num_components_after=0,
            components_reconnected=0,
            components_unreachable=0,
        )

    G = nx.Graph()
    G.add_edges_from(visibility_graph)
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    num_components_before = len(components)

    if num_components_before <= 1:
        return BridgeReconnectionResult(
            original_graph=visibility_graph,
            reconnected_graph=visibility_graph,
            bridge_edges=[],
            bridge_similarities=[],
            num_components_before=1,
            num_components_after=1,
            components_reconnected=0,
            components_unreachable=0,
        )

    main_comp = components[0]
    main_cameras = sorted(main_comp)
    existing_edges = set(visibility_graph)

    sim_np = similarity_matrix.numpy()

    bridge_edges: List[Tuple[int, int]] = []
    bridge_similarities: List[float] = []
    components_reconnected = 0
    components_unreachable = 0

    for comp_idx, comp_nodes in enumerate(components[1:], start=1):
        if len(comp_nodes) < min_component_size:
            continue

        island_cameras = sorted(comp_nodes)

        # Find cross-component edges ranked by similarity.
        cross_edges: List[Tuple[int, int, float]] = []
        for i in island_cameras:
            for j in main_cameras:
                a, b = (i, j) if i < j else (j, i)
                sim = float(sim_np[a, b])
                if sim >= min_bridge_similarity:
                    cross_edges.append((a, b, sim))

        # Sort by similarity descending, take top-K.
        cross_edges.sort(key=lambda e: -e[2])
        selected = cross_edges[:top_k_per_component]

        if len(selected) == 0:
            components_unreachable += 1
            logger.debug(
                "Island %d (%d cameras): no bridge edges above %.4f similarity.",
                comp_idx,
                len(comp_nodes),
                min_bridge_similarity,
            )
            continue

        components_reconnected += 1
        for a, b, sim in selected:
            if (a, b) not in existing_edges:
                bridge_edges.append((a, b))
                bridge_similarities.append(sim)
                existing_edges.add((a, b))
                logger.debug("Bridge edge (%d, %d) sim=%.4f [island %d]", a, b, sim, comp_idx)

        logger.info(
            "Island %d (%d cameras): added %d bridge edges (best sim=%.4f).",
            comp_idx,
            len(comp_nodes),
            len(selected),
            selected[0][2],
        )

    # Build reconnected graph.
    reconnected_graph = list(visibility_graph) + bridge_edges

    # Compute components after reconnection.
    G_after = nx.Graph()
    G_after.add_edges_from(reconnected_graph)
    num_components_after = nx.number_connected_components(G_after)

    return BridgeReconnectionResult(
        original_graph=visibility_graph,
        reconnected_graph=reconnected_graph,
        bridge_edges=bridge_edges,
        bridge_similarities=bridge_similarities,
        num_components_before=num_components_before,
        num_components_after=num_components_after,
        components_reconnected=components_reconnected,
        components_unreachable=components_unreachable,
    )
