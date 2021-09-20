"""
A library for cycle triplet extraction and cycle error computation.

Checks the cumulative rotation errors between triplets to throw away cameras.
Note: the same property does not hold for cumulative translation errors when scale is unknown (i.e. in SfM).

Author: John Lambert
"""

import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gtsam import Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.two_view_estimator import TwoViewEstimationReport


logger = logger_utils.get_logger()


CYCLE_ERROR_THRESHOLD = 5.0

MAX_INLIER_MEASUREMENT_ERROR_DEG = 5.0


def extract_triplets(i2Ri1_dict: Dict[Tuple[int, int], Rot3]) -> List[Tuple[int, int, int]]:
    """Discover triplets from a graph, without O(n^3) complexity, by using intersection within adjacency lists.

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
    adj_list = create_adjacency_list(i2Ri1_dict)

    # only want to keep the unique ones
    triplets = set()

    # find intersections
    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None:
            continue

        if i1 >= i2:
            raise RuntimeError("Graph edges (i1,i2) must be ordered with i1 < i2 in the image loader.")

        nodes_from_i1 = adj_list[i1]
        nodes_from_i2 = adj_list[i2]
        node_intersection = (nodes_from_i1).intersection(nodes_from_i2)
        for node in node_intersection:
            cycle_nodes = tuple(sorted([i1, i2, node]))
            if cycle_nodes not in triplets:
                triplets.add(cycle_nodes)

    return list(triplets)


def create_adjacency_list(i2Ri1_dict: Dict[Tuple[int, int], Rot3]) -> DefaultDict[int, Set[int]]:
    """Create an adjacency-list representation of a **rotation** graph G=(V,E) when provided its edges E.

    Note: this is specific to the rotation averaging use case, where some edges may be unestimated
    (i.e. their relative rotation is None), in which case they are not incorporated into the graph.

    In an adjacency list, the neighbors of each vertex may be listed efficiently, in time proportional to the
    degree of the vertex. In an adjacency matrix, this operation takes time proportional to the number of
    vertices in the graph, which may be significantly higher than the degree.

    Args:
        i2Ri1_dict: mapping from image pair indices to relative rotation.

    Returns:
        adj_list: adjacency list representation of the graph, mapping an image index to its neighbors
    """
    adj_list = defaultdict(set)

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None:
            continue

        adj_list[i1].add(i2)
        adj_list[i2].add(i1)

    return adj_list


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
        two_view_reports_dict: mapping from image pair indices (i1,i2) to a report containing information
            about the verifier's output (and optionally measurement error w.r.t GT). Note: i1 < i2 always.
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
        max_rot_error = float(np.max(rot_errors))
        max_trans_error = float(np.max(trans_errors))
    else:
        # ground truth unknown, so cannot estimate error w.r.t. GT
        max_rot_error = None
        max_trans_error = None

    if verbose:
        # for each rotation R: find a vector [x,y,z] s.t. R = Rot3.RzRyRx(x,y,z)
        # this is equivalent to scipy.spatial.transform's `.as_euler("xyz")`
        i1Ri0_euler = np.rad2deg(i1Ri0.xyz())
        i2Ri1_euler = np.rad2deg(i2Ri1.xyz())
        i0Ri2_euler = np.rad2deg(i0Ri2.xyz())

        logger.info("\n")
        logger.info(f"{i0},{i1},{i2} --> Cycle error is: {cycle_error:.1f}")
        if gt_known:
            logger.info(f"Triplet: w/ max. R err {max_rot_error:.1f}, and w/ max. t err {max_trans_error:.1f}")

        logger.info(
            "X: (0->1) %.1f deg., (1->2) %.1f deg., (2->0) %.1f deg.", i1Ri0_euler[0], i2Ri1_euler[0], i0Ri2_euler[0]
        )
        logger.info(
            "Y: (0->1) %.1f deg., (1->2) %.1f deg., (2->0) %.1f deg.", i1Ri0_euler[1], i2Ri1_euler[1], i0Ri2_euler[1]
        )
        logger.info(
            "Z: (0->1) %.1f deg., (1->2) %.1f deg., (2->0) %.1f deg.", i1Ri0_euler[2], i2Ri1_euler[2], i0Ri2_euler[2]
        )

    return cycle_error, max_rot_error, max_trans_error


def filter_to_cycle_consistent_edges(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    visualize: bool = True,
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3], GtsfmMetricsGroup]:
    """Remove edges in a graph where concatenated transformations along a 3-cycle does not compose to identity.

    Note: will return only a subset of these two dictionaries

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
        v_corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value as matching keypoint indices.
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

    triplets = extract_triplets(i2Ri1_dict)

    for (i0, i1, i2) in triplets:
        cycle_error, max_rot_error, max_trans_error = compute_cycle_error(
            i2Ri1_dict, (i0, i1, i2), two_view_reports_dict
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

    i2Ri1_dict_consistent, i2Ui1_dict_consistent, v_corr_idxs_dict_consistent = {}, {}, {}
    for (i1, i2) in cycle_consistent_keys:
        i2Ri1_dict_consistent[(i1, i2)] = i2Ri1_dict[(i1, i2)]
        i2Ui1_dict_consistent[(i1, i2)] = i2Ui1_dict[(i1, i2)]
        v_corr_idxs_dict_consistent[(i1, i2)] = v_corr_idxs_dict[(i1, i2)]

    logger.info("Found %d consistent rel. rotations from %d original edges.", len(i2Ri1_dict_consistent), n_valid_edges)

    metrics_group = _compute_metrics(
        inlier_i1_i2_pairs=cycle_consistent_keys, two_view_reports_dict=two_view_reports_dict
    )
    return i2Ri1_dict_consistent, i2Ui1_dict_consistent, v_corr_idxs_dict_consistent, metrics_group


def _compute_metrics(
    inlier_i1_i2_pairs: List[Tuple[int, int]], two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport]
) -> GtsfmMetricsGroup:
    """Computes the rotation cycle consistency metrics as a metrics group.

    Args:
        inlier_i1_i2_pairs: List of inlier camera pair indices.
        two_view_reports_dict: mapping from image pair indices (i1,i2) to a report containing information
            about the verifier's output (and optionally measurement error w.r.t GT). Note: i1 < i2 always.

    Returns:
        Rotation cycle consistency metrics as a metrics group. Includes the following metrics:
        - Number of inlier, outlier and total measurements.
        - Distribution of relative rotation angular errors for inlier measurements.
        - Distribution of relative rotation angular errors for outlier measurements.
        - Distribution of translation direction angular errors for inlier measurements.
        - Distribution of translation direction angular errors for outlier measurements.
    """
    all_pairs = list(two_view_reports_dict.keys())
    outlier_i1_i2_pairs = list(set(all_pairs) - set(inlier_i1_i2_pairs))
    num_total_measurements = len(all_pairs)

    inlier_R_angular_errors = []
    outlier_R_angular_errors = []

    inlier_U_angular_errors = []
    outlier_U_angular_errors = []

    for (i1, i2), report in two_view_reports_dict.items():

        if report.R_error_deg is None or report.U_error_deg is None:
            continue
        if (i1, i2) in inlier_i1_i2_pairs:
            inlier_R_angular_errors.append(report.R_error_deg)
            inlier_U_angular_errors.append(report.U_error_deg)
        else:
            outlier_R_angular_errors.append(report.R_error_deg)
            outlier_U_angular_errors.append(report.U_error_deg)

    R_precision, R_recall = metrics_utils.get_precision_recall_from_errors(
        inlier_R_angular_errors, outlier_R_angular_errors, MAX_INLIER_MEASUREMENT_ERROR_DEG
    )

    U_precision, U_recall = metrics_utils.get_precision_recall_from_errors(
        inlier_U_angular_errors, outlier_U_angular_errors, MAX_INLIER_MEASUREMENT_ERROR_DEG
    )

    rcc_metrics = [
        GtsfmMetric("num_total_frontend_measurements", num_total_measurements),
        GtsfmMetric("num_inlier_rcc_measurements", len(inlier_i1_i2_pairs)),
        GtsfmMetric("num_outlier_rcc_measurements", len(outlier_i1_i2_pairs)),
        GtsfmMetric("rot_cycle_consistency_R_precision", R_precision),
        GtsfmMetric("rot_cycle_consistency_R_recall", R_recall),
        GtsfmMetric("rot_cycle_consistency_U_precision", U_precision),
        GtsfmMetric("rot_cycle_consistency_U_recall", U_recall),
        GtsfmMetric("inlier_R_angular_errors_deg", inlier_R_angular_errors),
        GtsfmMetric("outlier_R_angular_errors_deg", outlier_R_angular_errors),
        GtsfmMetric("inlier_U_angular_errors_deg", inlier_U_angular_errors),
        GtsfmMetric("outlier_U_angular_errors_deg", outlier_U_angular_errors),
    ]

    return GtsfmMetricsGroup("rotation_cycle_consistency_metrics", rcc_metrics)
