"""
Utilities for cycle triplet extraction and cycle error computation.

In short, one can check the cumulative rotation errors between triplets to throw away cameras.
Note: the same property does not hold for cumulative translation errors when scale is unknown (i.e. in SfM).

Author: John Lambert
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gtsam import Rot3, Unit3
from scipy.spatial.transform import Rotation

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.two_view_estimator import TwoViewEstimationReport


logger = logger_utils.get_logger()


CYCLE_ERROR_THRESHOLD = 5.0


def extract_triplets_adjacency_list_intersection(i2Ri1_dict: Dict[Tuple[int, int], Rot3]) -> List[Tuple[int, int, int]]:
    """Discover triplets from a graph, without O(n^3) complexity.

    Based off of Theia's implementation:
        https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/math/graph/triplet_extractor.h

    If we have an edge a<->b, if we can find any node c such that a<->c and b<->c, then we have
    discovered a triplet. In other words, we need only look at the intersection between the nodes
    connected to `a` and the nodes connected to `b`.

    Args:
        i2Ri1_dict: mapping from image pair indices to relative rotation.

    Returns:
        triplets: 3-tuples of nodes that form a cycle. Nodes of each triplet are provided in sorted order.
    """
    # only want to keep the unique ones
    triplets = set()

    # form adjacency list
    adj_list = defaultdict(set)

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None:
            continue

        adj_list[i1].add(i2)
        adj_list[i2].add(i1)

    # find intersections
    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None:
            continue

        nodes_from_i1 = adj_list[i1]
        nodes_from_i2 = adj_list[i2]
        node_intersection = (nodes_from_i1).intersection(nodes_from_i2)
        for node in node_intersection:
            cycle_nodes = tuple(sorted([i1, i2, node]))
            if cycle_nodes not in triplets:
                triplets.add(cycle_nodes)

    return list(triplets)


def compute_cycle_error(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    cycle_nodes: Tuple[int, int, int],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    verbose: bool = True,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Compute the cycle error by the magnitude of the axis-angle rotation after composing 3 rotations.

    Note: a < b for every valid edge (a,b), by construction inside the image loader class.

    Args:
        i2Ri1_dict: mapping from image pair indices to relative rotation.
        cycle_nodes: 3-tuples of nodes that form a cycle. Nodes of are provided in sorted order.
        two_view_reports_dict:
        verbose: whether to dump to logger information about error in each Euler angle

    Returns:
        cycle_error: deviation from 3x3 identity matrix, in degrees. In other words,
            it is defined as the magnitude of the axis-angle rotation of the composed transformations.
        max_rot_error: maximum rotation error w.r.t. GT across triplet edges, in degrees.
            If ground truth is not known for a scene, None will be returned instead.
        max_trans_error: maximum translation error w.r.t. GT across triplet edges, in degrees.
            If ground truth is not known for a scene, None will be returned instead.
    """
    cycle_nodes = list(cycle_nodes)
    cycle_nodes.sort()

    i0, i1, i2 = cycle_nodes

    i1Ri0 = i2Ri1_dict[(i0, i1)]
    i2Ri1 = i2Ri1_dict[(i1, i2)]
    i0Ri2 = i2Ri1_dict[(i0, i2)].inverse()

    # should compose to identity, with ideal measurements
    i0Ri0 = i0Ri2.compose(i2Ri1).compose(i1Ri0)

    I_3x3 = Rot3()
    cycle_error = comp_utils.compute_relative_rotation_angle(I_3x3, i0Ri0)

    # form 3 edges e_i, e_j, e_k between fully connected subgraph (nodes i0,i1,i2)
    edges = [(i0, i1), (i1, i2), (i0, i2)]

    rot_errors = [two_view_reports_dict[e].R_error_deg for e in edges]
    trans_errors = [two_view_reports_dict[e].U_error_deg for e in edges]

    gt_known = all([err is not None for err in rot_errors])
    if gt_known:
        max_rot_error = np.max(rot_errors)
        max_trans_error = np.max(trans_errors)
    else:
        # ground truth unknown, so cannot estimate error w.r.t. GT
        max_rot_error = None
        max_trans_error = None

    if verbose:
        i1Ri0_euler = Rotation.from_matrix(i1Ri0.matrix()).as_euler(seq="xyz", degrees=True).tolist()
        i2Ri1_euler = Rotation.from_matrix(i2Ri1.matrix()).as_euler(seq="xyz", degrees=True).tolist()
        i0Ri2_euler = Rotation.from_matrix(i0Ri2.matrix()).as_euler(seq="xyz", degrees=True).tolist()

        euler_x = [i1Ri0_euler[0], i2Ri1_euler[0], i0Ri2_euler[0]]
        euler_y = [i1Ri0_euler[1], i2Ri1_euler[1], i0Ri2_euler[1]]
        euler_z = [i1Ri0_euler[2], i2Ri1_euler[2], i0Ri2_euler[2]]

        logger.info("\n")
        logger.info(f"{i0},{i1},{i2} --> Cycle error is: {cycle_error:.1f}")
        if gt_known:
            logger.info(f"Triplet: w/ max. R err {max_rot_error:.1f}, and w/ max. t err {max_trans_error:.1f}")

        logger.info("X: (0->1) %.1f deg., (1->2) %.1f deg., (2->0) %.1f deg.", euler_x[0], euler_x[1], euler_x[2])
        logger.info("Y: (0->1) %.1f deg., (1->2) %.1f deg., (2->0) %.1f deg.", euler_y[0], euler_y[1], euler_y[2])
        logger.info("Z: (0->1) %.1f deg., (1->2) %.1f deg., (2->0) %.1f deg.", euler_z[0], euler_z[1], euler_z[2])

    return cycle_error, max_rot_error, max_trans_error


def filter_to_cycle_consistent_edges(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    visualize: bool = True,
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """Remove edges in a graph where concatenated transformations along a 3-cycle does not compose to identity.

    Note: Will return only a subset of these two dictionaries

    Concatenating the transformations along a loop in the graph should return the identity function in an
    ideal, noise-free setting.

    Based off of:
        https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/filter_view_graph_cycles_by_rotation.cc

    See also:
        C. Zach, M. Klopschitz, and M. Pollefeys. Disambiguating visual relations using loop constraints. In CVPR, 2010
        http://people.inf.ethz.ch/pomarc/pubs/ZachCVPR10.pdf

        Enqvist, Olof; Kahl, Fredrik; Olsson, Carl. Non-Sequential Structure from Motion. ICCVW, 2011.
        https://portal.research.lu.se/ws/files/6239297/2255278.pdf

    Args:
        i2Ri1_dict: mapping from image pair indices (i1,i2) to relative rotation i2Ri1.
        i2Ui1_dict: mapping from image pair indices (i1,i2) to relative translation direction i2Ui1.
            Should have same keys as i2Ri1_dict.
        two_view_reports_dict: mapping from image pair indices (i1,i2) to a report containing information
            about the verifier's output (and optionally measurement error w.r.t GT). Note: i1 < i2 always.
        visualize: boolean indicating whether to plot cycle error vs. pose error w.r.t. GT

    Returns:
        i2Ri1_dict_consistent: subset of i2Ri1_dict, i.e. only including edges that belonged to some triplet
            and had cycle error below the predefined threshold.
        i2Ui1_dict_consistent: subset of i2Ui1_dict, as above.
    """
    cycle_errors = []
    max_rot_errors = []
    max_trans_errors = []

    n_valid_edges = len([i2Ri1 for (i1, i2), i2Ri1 in i2Ri1_dict.items() if i2Ri1 is not None])

    # (i1,i2) pairs
    cycle_consistent_keys = set()

    triplets = extract_triplets_adjacency_list_intersection(i2Ri1_dict)

    for (i0, i1, i2) in triplets:
        cycle_error, max_rot_error, max_trans_error = compute_cycle_error(
            i2Ri1_dict, [i0, i1, i2], two_view_reports_dict
        )

        if cycle_error < CYCLE_ERROR_THRESHOLD:
            # since i0 < i1 < i2 by construction, we preserve the property `a < b` for each edge (a,b)
            cycle_consistent_keys.add((i0, i1))
            cycle_consistent_keys.add((i1, i2))
            cycle_consistent_keys.add((i0, i2))

        cycle_errors.append(cycle_error)
        max_rot_errors.append(max_rot_error)
        max_trans_errors.append(max_trans_error)

    if visualize:
        plt.scatter(cycle_errors, max_rot_errors)
        plt.xlabel("Cycle error")
        plt.ylabel("Avg. Rot3 error over cycle triplet")
        plt.savefig(os.path.join("plots", "cycle_error_vs_GT_rot_error.jpg"), dpi=200)
        plt.close("all")

        plt.scatter(cycle_errors, max_trans_errors)
        plt.xlabel("Cycle error")
        plt.ylabel("Avg. Unit3 error over cycle triplet")
        plt.savefig(os.path.join("plots", "cycle_error_vs_GT_trans_error.jpg"), dpi=200)

    logger.info("cycle_consistent_keys: " + str(cycle_consistent_keys))

    i2Ri1_dict_consistent, i2Ui1_dict_consistent = {}, {}
    for (i1, i2) in cycle_consistent_keys:
        i2Ri1_dict_consistent[(i1, i2)] = i2Ri1_dict[(i1, i2)]
        i2Ui1_dict_consistent[(i1, i2)] = i2Ui1_dict[(i1, i2)]

    logger.info("Found %d consistent rel. rotations from %d original edges.", len(i2Ri1_dict_consistent), n_valid_edges)
    return i2Ri1_dict_consistent, i2Ui1_dict_consistent
