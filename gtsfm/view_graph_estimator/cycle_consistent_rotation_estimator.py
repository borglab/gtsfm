"""A ViewGraphEstimator implementation which ensures relative rotations are consistent in the cycles of the graph.

Authors: John Lambert, Ayush Baid
"""
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Set, Tuple

import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase

logger = logger_utils.get_logger()

ERROR_THRESHOLD = 7.0

MAX_INLIER_MEASUREMENT_ERROR_DEG = 5.0


class EdgeErrorAggregationCriterion(str, Enum):
    """Aggregate cycle errors over each edge by choosing one of the following summary statistics:

    MIN: Choose the mininum cycle error of all cyles this edge appears in. An edge that appears in ANY cycle
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
        i2Ri1: Dict[Tuple[int, int], Rot3],
        i2Ui1: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        keypoints: List[Keypoints],
    ) -> Set[Tuple[int, int]]:
        # pylint: disable=unused-argument

        logger.info("Input number of edges: %d" % len(i2Ri1))
        input_edges: List[Tuple[int, int]] = self.__get_valid_input_edges(i2Ri1)
        triplets: List[Tuple[int, int, int]] = graph_utils.extract_cyclic_triplets_from_edges(input_edges)

        logger.info("Number of triplets: %d" % len(triplets))

        per_edge_errors = defaultdict(list)
        cycle_errors: List[float] = []
        # Compute the cycle error for each triplet, and add it to its edges for aggregation.
        for i0, i1, i2 in triplets:  # sort order guaranteed
            error = comp_utils.compute_cyclic_rotation_error(
                i1Ri0=i2Ri1[(i0, i1)], i2Ri1=i2Ri1[(i1, i2)], i2Ri0=i2Ri1[(i0, i2)]
            )
            cycle_errors.append(error)
            per_edge_errors[(i0, i1)].append(error)
            per_edge_errors[(i1, i2)].append(error)
            per_edge_errors[(i0, i2)].append(error)

        # Filter the edges based on the aggregate error.
        per_edge_aggregate_error = {
            pair_indices: self.__aggregate_errors_for_edge(errors) for pair_indices, errors in per_edge_errors.items()
        }
        valid_edges = {edge for edge, error in per_edge_aggregate_error.items() if error < self._error_threshold}

        return valid_edges

    def __get_valid_input_edges(self, i2Ri1: Dict[Tuple[int, int], Rot3]) -> List[Tuple[int, int]]:
        """Gets the input edges (i1, i2) with the relative rotation i2Ri1 where:
        1. i1 < i2
        2. i2Ri1 is not None

        Args:
            i2Ri1: input dictionary of relative rotations.

        Returns:
            List of valid edges.
        """
        valid_edges = []
        for (i1, i2), i2Ri1 in i2Ri1.items():
            if i2Ri1 is None or i1 >= i2:
                logger.error("Incorrectly ordered edge indices found in cycle consistency for ({i1}, {i2})")
                continue
            else:
                valid_edges.append((i1, i2))

        return valid_edges

    def __aggregate_errors_for_edge(self, edge_errors: List[float]) -> float:
        """Aggregates a list of errors from different triplets into a single scalar value.

        The aggregation criterion is decided by `edge_error_aggregation_criterion`:
        MIN_EDGE_ERROR: Returns the minimum value among the input values.
        MEDIAN_EDGE_ERROR: Returns the median value among the input values.

        Args:
            errors (List[float]): A list of errors for the edge in different triplets.

        Returns:
            float: The aggregated error value.
        """
        if self._edge_error_aggregation_criterion == EdgeErrorAggregationCriterion.MIN_EDGE_ERROR:
            return np.amin(edge_errors)
        elif self._edge_error_aggregation_criterion == EdgeErrorAggregationCriterion.MEDIAN_EDGE_ERROR:
            return np.median(edge_errors)
