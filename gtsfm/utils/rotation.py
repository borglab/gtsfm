"""Utility functions for rotations.

Authors: Ayush Baid
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from gtsam import Rot3


def random_rotation(angle_scale_factor: float = 0.1) -> Rot3:
    """Sample a random rotation by generating a sample from the 4d unit sphere."""
    q = np.random.rand(4)
    # make unit-length quaternion
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    R = Rot3(qw, qx, qy, qz)
    axis, angle = R.axisAngle()
    angle = angle * angle_scale_factor
    return Rot3.AxisAngle(axis.point3(), angle)


def initialize_global_rotations_using_mst(
    num_images: int, i2Ri1_dict: Dict[Tuple[int, int], Rot3], edge_weights: Dict[Tuple[int, int], int]
) -> List[Rot3]:
    """Initializes rotations using minimum spanning tree (weighted by number of correspondences).

    Args:
        num_images: Number of images in the scene.
        i2Ri1_dict: Dictionary of relative rotations (i1, i2): i2Ri1.
        edge_weights: Weight of the edges (i1, i2). All edges in i2Ri1 must have an edge weight.

    Returns:
        Global rotations wRi initialized using an MST. Randomly initialized if we have a forest.
    """
    # Create a graph from the relative rotations dictionary.
    graph = nx.Graph()
    for i1, i2 in i2Ri1_dict.keys():
        graph.add_edge(i1, i2, weight=edge_weights[(i1, i2)])

    if not nx.is_connected(graph):
        raise ValueError("Relative rotation graph is not connected")

    # Compute the Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(graph)

    # MST graph.
    G = nx.Graph()
    G.add_edges_from(mst.edges)

    wRi_list: List[Rot3] = [Rot3()] * num_images
    # Choose origin node.
    origin_node = list(G.nodes)[0]
    wRi_list[origin_node] = Rot3()

    # Ignore 0th node, as we already set its global pose as the origin
    for dst_node in list(G.nodes)[1:]:
        # Determine the path to this node from the origin. ordered from [origin_node,...,dst_node]
        path = nx.shortest_path(G, source=origin_node, target=dst_node)

        # Chain relative rotations w.r.t. origin node. Initialize as identity Rot3 w.r.t origin node `i1`.
        wRi1 = Rot3()
        for i1, i2 in zip(path[:-1], path[1:]):
            # NOTE: i1, i2 may not be in sorted order here. May need to reverse ordering.
            if i1 < i2:
                i1Ri2 = i2Ri1_dict[(i1, i2)].inverse()
            else:
                i1Ri2 = i2Ri1_dict[(i2, i1)]
            # Path order is (origin -> ... -> i1 -> i2 -> ... -> dst_node). Set `i2` to be new `i1`.
            wRi1 = wRi1 * i1Ri2

        wRi_list[dst_node] = wRi1

    return wRi_list
