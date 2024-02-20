"""Utility functions for rotations.

Authors: Ayush Baid
"""
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from gtsam import Rot3


def random_rotation() -> Rot3:
    """Sample a random rotation by generating a sample from the 4d unit sphere."""
    q = np.random.randn(4) * 0.03
    # make unit-length quaternion
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    return Rot3(qw, qx, qy, qz)


def initialize_global_rotations_using_mst(num_images: int, i2Ri1_dict: Dict[Tuple[int, int], Rot3]) -> List[Rot3]:
        num_images: Number of images in the scene.
    # Create a graph from the relative rotations dictionary
    graph = nx.Graph()
    for i1, i2 in i2Ri1_dict.keys():
        # TODO: use inlier count as weight
        graph.add_edge(i1, i2, weight=1)

    # Compute the Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(graph)

    wRis = [random_rotation() for _ in range(num_images)]
    for i1, i2 in sorted(mst.edges):
        if (i1, i2) in i2Ri1_dict:
            wRis[i2] = wRis[i1] * i2Ri1_dict[(i1, i2)].inverse()
        else:
            wRis[i2] = wRis[i1] * i2Ri1_dict[(i2, i1)]

    return wRis
