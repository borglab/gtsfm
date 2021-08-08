"""Utilities for performing graph operations.

Authors: Ayush Baid
"""
from typing import Dict, List, Optional, Tuple

import networkx as nx
from gtsam import Rot3, Unit3


def get_nodes_in_largest_connected_component(edges: List[Tuple[int, int]]) -> List[int]:
    """Finds the nodes in the largest connected component of the bidirectional graph defined by the input edges.

    Args:
        edges: edges of the bi-directional graph.

    Returns:
        Nodes in the largest connected component of the input graph.
    """
    if len(edges) == 0:
        return []

    input_graph = nx.Graph()
    input_graph.add_edges_from(edges)

    # get the largest connected component
    largest_cc = max(nx.connected_components(input_graph), key=len)
    subgraph = input_graph.subgraph(largest_cc).copy()

    return list(subgraph.nodes())


def prune_to_largest_connected_component(
    rotations: Dict[Tuple[int, int], Optional[Rot3]], unit_translations: Dict[Tuple[int, int], Optional[Unit3]],
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """Process the graph of image indices with Rot3s/Unit3s defining edges, and select the largest connected component.

    Args:
        rotations: dictionary of relative rotations for pairs.
        unit_translations: dictionary of relative unit-translations for pairs.

    Returns:
        Subset of rotations which are in the largest connected components.
        Subset of unit_translations which are in the largest connected components.
    """
    input_edges = [k for (k, v) in rotations.items() if v is not None]
    nodes_in_pruned_graph = get_nodes_in_largest_connected_component(input_edges)

    # select the edges with nodes in the pruned graph
    selected_edges = []
    for i1, i2 in rotations.keys():
        if i1 in nodes_in_pruned_graph and i2 in nodes_in_pruned_graph:
            selected_edges.append((i1, i2))

    # return the subset of original input
    return (
        {k: rotations[k] for k in selected_edges},
        {k: unit_translations[k] for k in selected_edges},
    )