"""Utilities for performing graph operations.

Authors: Ayush Baid, John Lambert, Akshay Krishnan
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
    rotations: Dict[Tuple[int, int], Optional[Rot3]],
    unit_translations: Dict[Tuple[int, int], Optional[Unit3]],
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


def create_adjacency_list(edges: Tuple[int, int]) -> DefaultDict[int, Set[int]]:
    """Create an adjacency-list representation of a graph G=(V,E) when provided its edges E.

    In an adjacency list, the neighbors of each vertex may be listed efficiently, in time proportional to the
    degree of the vertex. In an adjacency matrix, this operation takes time proportional to the number of
    vertices in the graph, which may be significantly higher than the degree.

    Args:
        edges: indices of edges in the graph as a list of tuples.

    Returns:
        adj_list: adjacency list representation of the graph, mapping an image index to its neighbors
    """
    adj_list = defaultdict(set)

    for (a, b) in edges:
        adj_list[a].add(b)
        adj_list[b].add(a)

    return adj_list


def extract_cyclic_triplets_from_edges(edges: Tuple[int, int]) -> List[Tuple[int, int, int]]:
    """Extracts triplets from a graph edges by using intersection within adjacency lists.

    Based on Theia's implementation:
        https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/math/graph/triplet_extractor.h

    If we have an edge a<->b, if we can find any node c such that a<->c and b<->c, then we have
    discovered a triplet. In other words, we need only look at the intersection between the nodes
    connected to `a` and the nodes connected to `b`.

    Args:
        edges: indices of edges in the graph as a list of tuples.

    Returns:
        triplets: 3-tuples of nodes that form a cycle. Nodes of each triplet are provided in sorted order.
    """
    adj_list = create_adjacency_list(i2Ri1_dict)

    # only want to keep the unique ones
    triplets = set()

    # find intersections
    for (a, b) in edges:
        if a > b:
            a, b = b, a

        nodes_from_a = adj_list[a]
        nodes_from_b = adj_list[b]
        node_intersection = (nodes_from_a).intersection(nodes_from_b)
        for node in node_intersection:
            cycle_nodes = tuple(sorted([a, b, node]))
            if cycle_nodes not in triplets:
                triplets.add(cycle_nodes)

    return list(triplets)
