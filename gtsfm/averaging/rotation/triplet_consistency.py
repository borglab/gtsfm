from typing import DefaultDict, Dict, List, Optional, Set, Tuple
from gtsam import Rot3, Unit3
from collections import defaultdict
import os
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.two_view_estimator import TwoViewEstimationReport

def create_adjacency_list(i2Ri1_dict: Dict[Tuple[int, int], Rot3], i2Ui1_dict: Dict[Tuple[int, int], Unit3]) -> DefaultDict[int, Set[int]]:
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
        if i2Ri1 is None or ((i1, i2) not in i2Ui1_dict or i2Ui1_dict[(i1, i2)] is None):
            continue

        adj_list[i1].add(i2)
        adj_list[i2].add(i1)

    return adj_list


def extract_triplets(i2Ri1_dict: Dict[Tuple[int, int], Rot3], i2Ui1_dict: Dict[Tuple[int, int], Unit3]) -> List[Tuple[int, int, int]]:
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
    adj_list = create_adjacency_list(i2Ri1_dict, i2Ui1_dict)

    # only want to keep the unique ones
    triplets = set()

    # find intersections
    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None or ((i1, i2) not in i2Ui1_dict or i2Ui1_dict[(i1, i2)] is None):
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


def compute_cycle_error(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    cycle_nodes: Tuple[int, int, int],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    verbose: bool = False,
) -> Tuple[float, Optional[float], Optional[float]]:
    cycle_nodes = list(cycle_nodes)
    cycle_nodes.sort()

    i0, i1, i2 = cycle_nodes

    # 1 --- 0
    #  \   /
    #    2

    i1Ri0 = i2Ri1_dict[(i0, i1)]
    i2Ri1 = i2Ri1_dict[(i1, i2)]
    i2Ri0 = i2Ri1_dict[(i0, i2)]
    i1Ui0 = i2Ui1_dict[(i0, i1)]
    i2Ui1 = i2Ui1_dict[(i1, i2)]
    i2Ui0 = i2Ui1_dict[(i0, i2)]

    # Compute plane containing cameras
    i2_plane_normal = np.cross(i2Ui0, i2Ui1)
    i1_plane_normal = i2Ri1.inverse() * i2_plane_normal
    cycle_error = i1_plane_normal.dot(i1Ui0.point3()) / np.linalg.norm(i1_plane_normal)
    cycle_error = np.rad2deg(np.arccos(cycle_error))

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

    return cycle_error, max_rot_error, max_trans_error


def filter_to_cycle_consistent_edges(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    visualize: bool = True,
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3], GtsfmMetricsGroup]:

    cycle_errors = []
    max_rot_errors = []
    max_t_errors = []

    n_valid_edges = len([i2Ri1 for (i1, i2), i2Ri1 in i2Ri1_dict.items() if (i2Ri1 is not None and (i1, i2) in i2Ui1_dict and i2Ui1_dict[(i1, i2)] is not None)])

    # (i1,i2) pairs
    cycle_consistent_keys = set()

    per_edge_errors = defaultdict(list)

    triplets = extract_triplets(i2Ri1_dict, i2Ui1_dict)

    for (i0, i1, i2) in triplets:
        cycle_error, max_rot_error, max_t_error = compute_cycle_error(i2Ri1_dict, i2Ui1_dict, (i0, i1, i2), two_view_reports_dict)
        # since i0 < i1 < i2 by construction, we preserve the property `a < b` for each edge (a,b)
        per_edge_errors[(i0, i1)].append(cycle_error)
        per_edge_errors[(i1, i2)].append(cycle_error)
        per_edge_errors[(i0, i2)].append(cycle_error)

        cycle_errors.append(cycle_error)
        max_rot_errors.append(max_rot_error)
        max_t_errors.append(max_rot_error)

    inlier_errors_aggregate = []
    inlier_errors_wrt_gt = []

    outlier_errors_aggregate = []
    outlier_errors_wrt_gt = []

    plt.close("all")
    # aggregate info over per edge_errors
    for (i1, i2), edge_cycle_errors in per_edge_errors.items():
        error_aggregate = np.median(edge_cycle_errors)

    if visualize:
        plt.scatter(
            inlier_errors_aggregate,
            inlier_errors_wrt_gt,
            10,
            color="g",
            marker=".",
            label=f"outliers @ {CYCLE_ERROR_THRESHOLD} deg.",
        )
        plt.scatter(
            outlier_errors_aggregate,
            outlier_errors_wrt_gt,
            10,
            color="r",
            marker=".",
            label=f"inliers @ {CYCLE_ERROR_THRESHOLD} deg.",
        )
        plt.xlabel("cycle error")
        plt.ylabel("Rotation error w.r.t GT")
        plt.axis("equal")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join("plots", "gt_err_vs_agg_error.jpg"), dpi=400)
        plt.close("all")

        plt.scatter(cycle_errors, max_rot_errors)
        plt.xlabel("Cycle error")
        plt.ylabel("Avg. Rot3 error over cycle triplet")
        plt.axis("equal")
        plt.savefig(os.path.join("plots", "cycle_error_vs_GT_rot_error.jpg"), dpi=400)
        plt.close("all")

        plt.scatter(cycle_errors, max_t_errors)
        plt.xlabel("Cycle error")
        plt.ylabel("Avg. Rot3 error over cycle triplet")
        plt.axis("equal")
        plt.savefig(os.path.join("plots", "cycle_error_vs_GT_T_error.jpg"), dpi=400)
        plt.close("all")

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
