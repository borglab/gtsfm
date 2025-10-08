"""Utilities for performing graph operations.

Authors: Ayush Baid, John Lambert, Akshay Krishnan
"""

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gtsam import PinholeCameraCal3Bundler, Rot3, Unit3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.products.visibility_graph import AnnotatedGraph

logger = logger_utils.get_logger()

GREEN = [0, 1, 0]
RED = [1, 0, 0]


def get_nodes_in_largest_connected_component(edges: List[Tuple[int, int]]) -> List[int]:
    """Finds the nodes in the largest connected component of the bidirectional graph defined by the input edges.

    Args:
        edges: Edges of the bi-directional graph.

    Returns:
        Nodes in the largest connected component of the input graph.
    """
    if len(edges) == 0:
        return []

    input_graph = nx.Graph()
    input_graph.add_edges_from(edges)

    # Log the sizes of the connected components.
    cc_sizes = [len(x) for x in sorted(list(nx.connected_components(input_graph)))]
    logger.info("Connected component sizes: %s nodes.", str(cc_sizes))

    # Get the largest connected component.
    largest_cc_nodes = max(nx.connected_components(input_graph), key=len)
    subgraph = input_graph.subgraph(largest_cc_nodes).copy()

    return list(subgraph.nodes())


def prune_to_largest_connected_component(
    rotations: Dict[Tuple[int, int], Optional[Rot3]],
    unit_translations: Dict[Tuple[int, int], Optional[Unit3]],
    relative_pose_priors: Dict[Tuple[int, int], PosePrior],
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """Process the graph of image indices with Rot3s/Unit3s defining edges, and select the largest connected component.

    As edges in the rotations and unit_translations are the same, a combination of edges from rotations and pose_priors
    dict are considered.

    Args:
        rotations: Dictionary of relative rotations for pairs.
        unit_translations: Dictionary of relative unit-translations for pairs.
        pose_priors: Dictionary of priors on relative pose.

    Returns:
        Subset of rotations which are in the largest connected components.
        Subset of unit_translations which are in the largest connected components.
    """
    input_edges = [k for (k, v) in rotations.items() if v is not None]
    input_edges += relative_pose_priors.keys()
    nodes_in_pruned_graph = get_nodes_in_largest_connected_component(input_edges)

    # Select the edges with nodes in the pruned graph.
    selected_edges = []
    for i1, i2 in rotations.keys():
        if i1 in nodes_in_pruned_graph and i2 in nodes_in_pruned_graph:
            selected_edges.append((i1, i2))

    logger.info(
        "Pruned to largest connected component with %d nodes and %d edges.",
        len(nodes_in_pruned_graph),
        len(selected_edges),
    )

    # Return the subset of original input.
    return (
        {k: rotations[k] for k in selected_edges},
        {k: unit_translations[k] for k in selected_edges},
    )


def create_adjacency_list(edges: List[Tuple[int, int]]) -> DefaultDict[int, Set[int]]:
    """Create an adjacency-list representation of a graph G=(V,E) when provided its edges E.

    In an adjacency list, the neighbors of each vertex may be listed efficiently, in time proportional to the
    degree of the vertex. In an adjacency matrix, this operation takes time proportional to the number of
    vertices in the graph, which may be significantly higher than the degree.

    Args:
        edges: Indices of edges in the graph as a list of tuples.

    Returns:
        adj_list: Adjacency list representation of the graph, mapping an image index to its neighbors.
    """
    adj_list = defaultdict(set)

    for a, b in edges:
        adj_list[a].add(b)
        adj_list[b].add(a)

    return adj_list


def extract_cyclic_triplets_from_edges(edges: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """Extracts triplets from a graph's edges by using intersection within adjacency lists.

    Based off of Theia and OpenMVG's implementations:
        https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/math/graph/triplet_extractor.h
        https://github.com/openMVG/openMVG/blob/develop/src/openMVG/graph/triplet_finder.hpp

    If we have an edge a<->b, if we can find any node c such that a<->c and b<->c, then we have
    discovered a triplet. In other words, we need only look at the intersection between the nodes
    connected to `a` and the nodes connected to `b`.

    Args:
        edges: Indices of edges in the graph as a list of tuples.

    Returns:
        triplets: 3-tuples of nodes that form a cycle. Nodes of each triplet are provided in sorted order.
    """
    adj_list = create_adjacency_list(edges)

    # Only want to keep the unique ones.
    triplets = set()

    # find intersections
    for a, b in edges:
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


def draw_view_graph_topology(
    edges: List[Tuple[int, int]],
    two_view_reports: AnnotatedGraph[TwoViewEstimationReport],
    title: str,
    save_fpath: str,
    cameras_gt: Optional[List[PinholeCameraCal3Bundler]] = None,
) -> None:
    """Draw the topology of an undirected graph, with edges colored by error.

    Note: False positive edges are colored red, and true positive edges are colored green.
    If ground truth camera parameters are provided, vertices are placed in their ground truth locations.
    Otherwise, we allow networkx to choose the vertex locations (placed at arbitrary locations).

    Args:
        edges: List of (i1,i2) pairs.
        two_view_reports: Two-view estimation report per edge.
        title: Desired title of figure.
        save_fpath: File path where plot should be saved to disk.
        cameras_gt: Ground truth camera parameters (including their poses).
    """
    M = len(edges)

    plt.figure(figsize=(16, 10))
    G = nx.Graph()
    G.add_edges_from(edges)
    nodes = list(G.nodes)

    R_errors = np.array([two_view_reports[edge].R_error_deg for edge in edges]).astype(np.float32)
    U_errors = np.array([two_view_reports[edge].U_error_deg for edge in edges]).astype(np.float32)

    if np.isnan(R_errors).any() or np.isnan(U_errors).any():
        # cannot color by error, as GT is not available.
        edge_colors = [GREEN] * M
    else:
        pose_errors = np.maximum(R_errors, U_errors)
        edge_colors = [GREEN if pose_error < 5 else RED for pose_error in pose_errors]

    if cameras_gt is not None:
        node_positions = {i: cameras_gt[i].pose().translation()[:2] for i in nodes}
    else:
        node_positions = None

    nx.drawing.nx_pylab.draw_networkx(
        G,
        edgelist=edges,
        edge_color=edge_colors,
        pos=node_positions,
        arrows=True,
        with_labels=True,
    )
    plt.axis("equal")
    plt.title(title)

    plt.savefig(save_fpath, dpi=500)
    plt.close("all")
