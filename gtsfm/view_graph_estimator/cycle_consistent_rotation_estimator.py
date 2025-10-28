"""A ViewGraphEstimator implementation which ensures relative rotations are consistent in the 3-cycles of the graph.

Authors: John Lambert, Ayush Baid, Akshay Krishnan
"""

import os
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3  # type: ignore

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.products.visibility_graph import AnnotatedGraph, ImageIndexPairs
from gtsfm.two_view_estimator import TwoViewEstimationReport
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase

logger = logger_utils.get_logger()

# threshold for cycle consistency inference
ERROR_THRESHOLD = 7.0

# threshold for evaluation w.r.t. GT
MAX_INLIER_MEASUREMENT_ERROR_DEG = 5.0


class EdgeErrorAggregationCriterion(str, Enum):
    """Aggregate cycle errors over each edge by choosing one of the following summary statistics:

    MIN: Choose the minimum cycle error of all cycles this edge appears in. An edge that appears in ANY cycle
        with low error is accepted. High recall, but can have low precision, as false positives can enter
        (error was randomly cancelled out by another error, meaning accepted).
    MEDIAN: Choose the median cycle error. robust summary statistic. At least half of the time, this edge
        appears in a good cycle (i.e. of low error). Note: preferred over mean, which is not robust to outliers.

    Note: all summary statistics will be compared with an allowed upper bound/threshold. If they exceed the
    upper bound, they will be rejected.
    """

    MIN_EDGE_ERROR = "MIN_EDGE_ERROR"
    MEDIAN_EDGE_ERROR = "MEDIAN_EDGE_ERROR"


class CycleConsistentRotationViewGraphEstimator(ViewGraphEstimatorBase):
    """A ViewGraphEstimator that filters two-view edges with high rotation-cycle consistency error.

    The rotation cycle consistency error is computed by composing two-view relative rotations along a triplet, i.e:
        inv(i2Ri0) * i2Ri1 * i1Ri0, where i2Ri0, i2Ri1 and i1Ri0 are 3 two-view relative rotations.

    For each edge, the cyclic rotation error is computed for all 3-edge cycles (triplets) that it is a part of, which
    are later aggregated using an EdgeErrorAggregationCriterion into a single value. The edge is added into the
    ViewGraph is the aggregated error is less than a threshold.

    Based off of:
        https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/filter_view_graph_cycles_by_rotation.cc

    See also:
        C. Zach, M. Klopschitz, and M. Pollefeys. Disambiguating visual relations using loop constraints. In CVPR, 2010
        http://people.inf.ethz.ch/pomarc/pubs/ZachCVPR10.pdf

        Enqvist, Olof; Kahl, Fredrik; Olsson, Carl. Non-Sequential Structure from Motion. ICCVW, 2011.
        https://portal.research.lu.se/ws/files/6239297/2255278.pdf
    """

    def __init__(
        self,
        edge_error_aggregation_criterion: EdgeErrorAggregationCriterion,
        error_threshold: float = ERROR_THRESHOLD,
    ) -> None:
        self._edge_error_aggregation_criterion = edge_error_aggregation_criterion
        self._error_threshold = error_threshold

    def run(
        self,
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        keypoints: List[Keypoints],
        two_view_reports: AnnotatedGraph[TwoViewEstimationReport],
        output_dir: Optional[Path] = None,
    ) -> Set[Tuple[int, int]]:
        """Estimates the view graph using the rotation consistency constraint in a cycle of 3 edges.

        Args:
            i2Ri1_dict: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1_dict: Dict from (i1, i2) to relative translation direction of i1 with respect to i2 (unused).
            calibrations: List of calibrations for each image (unused).
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2 (unused).
            keypoints: keypoints for each images (unused).
            two_view_reports: Dict from (i1, i2) to the TwoViewEstimationReport of the edge.
            output_dir: Path to directory where outputs for debugging will be saved.

        Returns:
            Edges of the view-graph, which are the subset of the image pairs in the input args.
        """
        # pylint: disable=unused-argument
        start_time = time.time()

        logger.info("Input number of edges: %d" % len(i2Ri1_dict))
        input_edges: ImageIndexPairs = list(i2Ri1_dict.keys())
        triplets: List[Tuple[int, int, int]] = graph_utils.extract_cyclic_triplets_from_edges(input_edges)

        logger.info("Number of triplets: %d" % len(triplets))

        per_edge_errors = defaultdict(list)
        cycle_errors: List[float] = []
        max_gt_error_in_cycle = []

        # Compute the cycle error for each triplet, and add it to its edges for aggregation.
        for i0, i1, i2 in triplets:  # sort order guaranteed
            error = comp_utils.compute_cyclic_rotation_error(
                i1Ri0=i2Ri1_dict[(i0, i1)], i2Ri1=i2Ri1_dict[(i1, i2)], i2Ri0=i2Ri1_dict[(i0, i2)]
            )
            cycle_errors.append(error)
            per_edge_errors[(i0, i1)].append(error)
            per_edge_errors[(i1, i2)].append(error)
            per_edge_errors[(i0, i2)].append(error)

            # Form 3 edges e_i, e_j, e_k between fully connected subgraph (nodes i0,i1,i2).
            edges = [(i0, i1), (i1, i2), (i0, i2)]
            rot_errors = [two_view_reports[e].R_error_deg for e in edges]
            gt_known = all([err is not None for err in rot_errors])
            # if ground truth unknown, cannot estimate error w.r.t. GT
            max_rot_error = max(rot_errors) if gt_known else None
            max_gt_error_in_cycle.append(max_rot_error)

        # Filter the edges based on the aggregate error.
        per_edge_aggregate_error = {
            pair_indices: self.__aggregate_errors_for_edge(errors) for pair_indices, errors in per_edge_errors.items()
        }
        valid_edges = {edge for edge, error in per_edge_aggregate_error.items() if error < self._error_threshold}
        # Add edges that were not part of a cycle.
        valid_edges.update({edge for edge in input_edges if edge not in per_edge_aggregate_error.keys()})

        if output_dir:
            self.__save_plots(
                valid_edges, cycle_errors, max_gt_error_in_cycle, per_edge_aggregate_error, two_view_reports, output_dir
            )

        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info(
            "Found %d consistent rel. rotations from %d original edges in %.2f sec.",
            len(valid_edges),
            len(input_edges),
            duration_sec,
        )

        return valid_edges

    def __save_plots(
        self,
        inlier_edges: Set[Tuple[int, int]],
        cycle_errors: List[float],
        max_gt_error_in_cycle: List[float],
        per_edge_aggregate_error: Dict[Tuple[int, int], float],
        two_view_reports_dict: AnnotatedGraph[TwoViewEstimationReport],
        output_dir: Path,
    ) -> None:
        """Saves plots of aggregate error vs GT error for each edge, and cyclic error vs max GT error for each cycle.

        Args:
            inlier_edges: Set of all cycle consistent edges.
            cycle_errors: Cyclic error for all cycles.
            max_gt_error_in_cycle: Maximum GT rotation error in the cycle, for all cycles.
            per_edge_aggregate_error: Dict from edge index pair to aggregate cyclic error of the edge.
            two_view_reports_dict: Dict from edge index pair to the TwoViewEstimationReport of the edge.
            output_dir: Path to directory where outputs for debugging will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        # Aggregate info over per edge_errors.
        inlier_errors_aggregate = []
        inlier_errors_wrt_gt = []

        outlier_errors_aggregate = []
        outlier_errors_wrt_gt = []
        for (i1, i2), error_aggregate in per_edge_aggregate_error.items():
            if (i1, i2) in inlier_edges:
                inlier_errors_aggregate.append(error_aggregate)
                inlier_errors_wrt_gt.append(two_view_reports_dict[(i1, i2)].R_error_deg)
            else:
                outlier_errors_aggregate.append(error_aggregate)
                outlier_errors_wrt_gt.append(two_view_reports_dict[(i1, i2)].R_error_deg)
        plt.scatter(
            inlier_errors_aggregate,
            inlier_errors_wrt_gt,
            10,
            color="g",
            marker=".",
            label=f"inliers @ {MAX_INLIER_MEASUREMENT_ERROR_DEG} deg.",
        )
        plt.scatter(
            outlier_errors_aggregate,
            outlier_errors_wrt_gt,
            10,
            color="r",
            marker=".",
            label=f"outliers @ {MAX_INLIER_MEASUREMENT_ERROR_DEG} deg.",
        )
        plt.xlabel(f"{self._edge_error_aggregation_criterion} cycle error")
        plt.ylabel("Rotation error w.r.t GT")
        plt.axis("equal")
        plt.legend(loc="lower right")
        plt.savefig(
            os.path.join(output_dir, f"gt_err_vs_{self._edge_error_aggregation_criterion}_agg_error.jpg"), dpi=400
        )
        plt.close("all")

        plt.scatter(cycle_errors, max_gt_error_in_cycle)
        plt.xlabel("Cycle error")
        plt.ylabel("Avg. Rot3 error over cycle triplet")
        plt.axis("equal")
        plt.savefig(os.path.join(output_dir, "cycle_error_vs_GT_rot_error.jpg"), dpi=400)
        plt.close("all")

    def __aggregate_errors_for_edge(self, edge_errors: List[float]) -> float:
        """Aggregates a list of errors from different triplets into a single scalar value.

        The aggregation criterion is decided by `edge_error_aggregation_criterion`:
        MIN_EDGE_ERROR: Returns the minimum value among the input values.
        MEDIAN_EDGE_ERROR: Returns the median value among the input values.

        Args:
            edge_errors: A list of errors for the edge in different triplets.

        Returns:
            float: The aggregated error value.
        """
        if self._edge_error_aggregation_criterion == EdgeErrorAggregationCriterion.MIN_EDGE_ERROR:
            return np.amin(edge_errors)
        elif self._edge_error_aggregation_criterion == EdgeErrorAggregationCriterion.MEDIAN_EDGE_ERROR:
            return np.median(edge_errors)
