"""Utilities for performing graph operations.

Authors: Ayush Baid
"""
from typing import List, Tuple

import networkx as nx


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

    # get the largest connected components and cameras in it
    largest_cc = max(nx.connected_components(input_graph), key=len)
    subgraph = input_graph.subgraph(largest_cc).copy()

    return list(subgraph.nodes())
