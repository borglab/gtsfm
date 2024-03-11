"""Utility functions for rotations.

Authors: Ayush Baid
"""

from typing import Dict, List, Tuple

import networkx as nx
from gtsam import Rot3


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

    wRis = [Rot3() for _ in range(num_images)]
    for i1, i2 in sorted(mst.edges):
        # NOTE: i1, i2 may not be in sorted order here. May need to reverse ordering.
        if (i1, i2) in i2Ri1_dict:
            i1Ri2 = i2Ri1_dict[(i1, i2)].inverse()
        else:
            i1Ri2 = i2Ri1_dict[(i2, i1)]
        wRis[i2] = wRis[i1] * i1Ri2

    return wRis



def initialize_mst(
        num_images: int, 
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]], 
        corr_idxs: Dict[Tuple[int, int], np.ndarray],
        old_to_new_idxs: Dict[int, int],
    ) -> gtsam.Values:
    """Initialize global rotations using the minimum spanning tree (MST)."""
    # Compute MST.
    row, col, data = [], [], []
    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None:
            continue
        row.append(i1)
        col.append(i2)
        data.append(-corr_idxs[(i1, i2)].shape[0])
    logger.info(corr_idxs[(i1, i2)])
    corr_adjacency = scipy.sparse.coo_array((data, (row, col)), shape=(num_images, num_images))
    Tcsr = scipy.sparse.csgraph.minimum_spanning_tree(corr_adjacency)
    logger.info(Tcsr.toarray().astype(int))

    # Build global rotations from MST.
    # TODO (travisdriver): This is simple but very inefficient. Use something else.
    i_mst, j_mst = Tcsr.nonzero()
    logger.info(i_mst)
    logger.info(j_mst)
    edges_mst = [(i, j) for (i, j) in zip(i_mst, j_mst)]
    iR0_dict = {i_mst[0]: np.eye(3)}  # pick the left index of the first edge as the seed
    # max_iters = num_images * 10
    iter = 0
    while len(edges_mst) > 0:
        i, j = edges_mst.pop(0)
        if i in iR0_dict:
            jRi = i2Ri1_dict[(i, j)].matrix()
            iR0 = iR0_dict[i]
            iR0_dict[j] = jRi @ iR0
        elif j in iR0_dict:
            iRj = i2Ri1_dict[(i, j)].matrix().T
            jR0 = iR0_dict[j]
            iR0_dict[i] = iRj @ jR0
        else:
            edges_mst.append((i, j))
        iter += 1
        # if iter >= max_iters:
        #     logger.info("Reached max MST iters.")
        #     assert False
    
    # Add to Values object.
    initial = gtsam.Values()
    for i, iR0 in iR0_dict.items():
        initial.insert(old_to_new_idxs[i], Rot3(iR0))
    
    return initial