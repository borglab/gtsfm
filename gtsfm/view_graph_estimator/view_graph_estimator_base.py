"""Implements a base class for ViewGraph estimation.

Estimating the ViewGraph can be done trivially by adding all the two-view estimates into a ViewGraph data structure.
The purpose of this class, however, is to define an API for more sophisticated methods for estimating a ViewGraph 
that include filtering or optimizing the two-view estimates.

Authors: Akshay Krishnan, Ayush Baid
"""
import abc
from logging import Logger
from typing import Dict, List, Optional, Set, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Rot3, Unit3

import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.two_view_estimator import TwoViewEstimationReport

MAX_INLIER_MEASUREMENT_ERROR_DEG = 5.0
METRIC_GROUP = "view_graph"

logger = logger_utils.get_logger()

class ViewGraphEstimatorBase(metaclass=abc.ABCMeta):
    """Base class for ViewGraph estimation.

    A ViewGraphEstimator aggregates two-view estimates into a ViewGraph.
    It could also improve the two-view estimates using filtering or optimization techniques.
    """

    @abc.abstractmethod
    def run(
        self,
        i2Ri1: Dict[Tuple[int, int], Rot3],
        i2Ui1: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        keypoints: List[Keypoints],
        two_view_reports: Dict[Tuple[int, int]: TwoViewEstimationReport],
    ) -> Set[Tuple[int, int]]:
        """Estimates the view graph, needs to be implemented by the derived class. 

        Args:
            i2Ri1: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            calibrations: list of calibrations for each image.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            keypoints: keypoints for each images.
            two_view_reports: two-view reports between image pairs from the TwoViewEstimator.

        Returns:
            Edges of the view-graph, which are the subset of the image pairs in the input args.
        """

    def __filter_with_edges(
        self,
        i2Ri1: Dict[Tuple[int, int], Rot3],
        i2Ui1: Dict[Tuple[int, int], Unit3],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
        edges_to_select: Set[Tuple[int, int]],
    ) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3], Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], TwoViewEstimationReport]]:
        """Filter the dictionaries of 2-view results with the image-pair edges.
        Args:
            i2Ri1: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            two_view_reports: two-view reports between image pairs from the TwoViewEstimator.
            edges_to_select: edges to select (tuple of image pair indices)

        Returns:
            Subset of i2Ri1.
            Subset of i2Ui1.
            Subset of corr_idxs_i1i2.
            Subset of two_view_reports.
        """
        return (
            {edge: i2Ri1[edge] for edge in edges_to_select},
            {edge: i2Ui1[edge] for edge in edges_to_select},
            {edge: corr_idxs_i1i2[edge] for edge in edges_to_select},
            {edge: two_view_reports[edge] for edge in edges_to_select}
        )

    def compute_metrics(
        self,
        i2Ri1: Dict[Tuple[int, int], Rot3],
        i2Ui1: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
        view_graph_edges: Tuple[int, int],
    ) -> GtsfmMetricsGroup:
        """Metric computation for the view optimizer by selecting a subset of two-view reports for the pairs which
        are the edges of the view-graph. This can be overrided by implementations to define custom metrics.

        Args:
            i2Ri1: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            calibrations: list of calibrations for each image.
            two_view_reports: two-view reports between image pairs from the TwoViewEstimator.
            view_graph_edges: edges of the view-graph.

        Returns:
            Metrics for the view graph estimation, as a GtsfmMetricsGroup.
        """
        # pylint: disable=unused-argument

        # Case of missing ground truth.
        if len(two_view_reports) == 0:
            return GtsfmMetricsGroup(name="rotation_cycle_consistency_metrics", metrics=[])

        input_edges = two_view_reports.keys()
        inlier_i1_i2 = view_graph_edges
        outlier_i1_i2 = [i1_i2 for i1_i2 in input_edges if i1_i2 not in inlier_i1_i2]

        inlier_R_angular_errors = []
        outlier_R_angular_errors = []

        inlier_U_angular_errors = []
        outlier_U_angular_errors = []

        for (i1, i2), report in two_view_reports.items():
            if report is None:
                logger.error('TwoViewEstimationReport is None for ({}, {})'.format(i1, i2))
            if (i1, i2) in inlier_i1_i2:
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
        view_graph_metrics = [
            GtsfmMetric("num_input_measurements", len(two_view_reports)),
            GtsfmMetric("num_inlier_measurements", len(inlier_i1_i2)),
            GtsfmMetric("num_outlier_measurements", len(outlier_i1_i2)),
            GtsfmMetric("R_precision", R_precision),
            GtsfmMetric("R_recall", R_recall),
            GtsfmMetric("U_precision", U_precision),
            GtsfmMetric("U_recall", U_recall),
            GtsfmMetric("inlier_R_angular_errors_deg", inlier_R_angular_errors),
            GtsfmMetric("outlier_R_angular_errors_deg", outlier_R_angular_errors),
            GtsfmMetric("inlier_U_angular_errors_deg", inlier_U_angular_errors),
            GtsfmMetric("outlier_U_angular_errors_deg", outlier_U_angular_errors),
        ]
        return GtsfmMetricsGroup("view_graph_estimation_metrics", view_graph_metrics)

    def create_computation_graph(
        self,
        i2Ri1: Delayed,
        i2Ui1: Delayed,
        calibrations: Delayed,
        corr_idxs_i1i2: Delayed,
        keypoints: Delayed,
        two_view_reports: Delayed,
    ) -> Tuple[Delayed, Delayed, Delayed, Delayed, Delayed]:
        """Create the computation graph for ViewGraph estimation and metric evaluation.
        
        Args:
            i2Ri1: Dict from (i1, i2) to relative rotation of i1 with respect to i2, wrapped as Delayed.
            i2Ui1: Dict from (i1, i2) to relative translation direction of i1 with respect to i2, wrapped as Delayed.
            calibrations: list of calibrations for each image, wrapped as Delayed.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2, wrapped as Delayed.
            keypoints: keypoints for each images, wrapped as Delayed.
            two_view_reports: Dict from (i1, i2) to TwoViewEstimationReport that contains metrics, wrapped as Delayed.

        Returns:
            Tuple of the following 5 elements, all wrapped as Delayed:
            - Dict of i2Ri1 in the view graph 
            - Dict of i2Ui1 in the view graph
            - Dict of corr_idxs_i1i2 in the view graph
            - Dict of two_view_reports in the view graph
            - GtsfmMetricsGroup with the view graph estimation metrics
        """
        view_graph_edges = dask.delayed(self.run)(i2Ri1, i2Ui1, calibrations, corr_idxs_i1i2, keypoints, two_view_reports)
        i2Ri1_filtered, i2Ui1_filtered, corr_idxs_i1i2_filtered, two_view_reports_filtered = dask.delayed(self.__filter_with_edges, nout=4)(
            i2Ri1, i2Ui1, corr_idxs_i1i2, two_view_reports, view_graph_edges
        )
        view_graph_estimation_metrics = dask.delayed(self.compute_metrics)(i2Ri1, i2Ui1, calibrations, two_view_reports, view_graph_edges)
        return i2Ri1_filtered, i2Ui1_filtered, corr_idxs_i1i2_filtered, two_view_reports_filtered, view_graph_estimation_metrics
