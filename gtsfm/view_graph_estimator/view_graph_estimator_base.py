"""Implements a base class for ViewGraph estimation.

Estimating the ViewGraph can be done trivially by adding all the two-view estimates into a ViewGraph data structure.
The purpose of this class, however, is to define an API for more sophisticated methods for estimating a ViewGraph 
that include filtering or optimizing the two-view estimates.

Authors: Akshay Krishnan, Ayush Baid
"""
import abc
from typing import Dict, List, Set, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Rot3, Unit3

from gtsfm.common.keypoints import Keypoints
from gtsfm.two_view_estimator import TwoViewEstimationReport

METRIC_GROUP = "view_graph"


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
    ) -> Set[Tuple[int, int]]:
        """Run the ViewGraph estimation.

        Args:
            i2Ri1: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            calibrations: list of calibrations for each image.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            keypoints: keypoints for each images.

        Returns:
            Edges of the view-graph, which are the subset of the image pairs in the input args.
        """

    def filter_with_edges(
        self,
        i2Ri1: Dict[Tuple[int, int], Rot3],
        i2Ui1: Dict[Tuple[int, int], Unit3],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        edges_to_select: Set[Tuple[int, int]],
    ) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3], Dict[Tuple[int, int], np.ndarray]]:
        """Filter the dictionaries of 2-view results with the image-pair edges.
        Args:
            i2Ri1: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            edges_to_select: edges to select (tuple of image pair indices)

        Returns:
            Subset of i2Ri1.
            Subset of i2Ui1.
            Subset of corr_idxs_i1i2.

        """

        return (
            {edge: i2Ri1[edge] for edge in edges_to_select},
            {edge: i2Ui1[edge] for edge in edges_to_select},
            {edge: corr_idxs_i1i2[edge] for edge in edges_to_select},
        )

    def compute_metrics(
        self, two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport], view_graph_edges: Set[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], TwoViewEstimationReport]:
        """Metric computation for the view optimizer by selecting a subset of two-view reports for the pairs which
        are the edges of the view-graph.

        Args:
            two_view_reports: two-view reports between image pairs from the TwoViewEstimator.
            view_graph_edges: edges of the view-graph.

        Returns:
            Subset of two_view_reports, only including the edges in view_graph_edges.
        """
        return {edge: two_view_reports[edge] for edge in view_graph_edges}

    def create_computation_graph(
        self,
        i2Ri1: Delayed,
        i2Ui1: Delayed,
        calibrations: Delayed,
        corr_idxs_i1i2: Delayed,
        keypoints: Delayed,
        two_view_reports: Dict[Tuple[int, int], Delayed],
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for ViewGraph estimation and metric evaluation."""
        view_graph_edges = dask.delayed(self.run, nout=2)(i2Ri1, i2Ui1, calibrations, corr_idxs_i1i2, keypoints)
        i2Ri1_filtered, i2Ui1_filtered, corr_idxs_i1i2_filtered = dask.delayed(self.filter_with_edges, nout=3)(
            i2Ri1, i2Ui1, corr_idxs_i1i2, view_graph_edges
        )
        two_view_reports_filtered = dask.delayed(self.compute_metrics)(two_view_reports, view_graph_edges)

        return i2Ri1_filtered, i2Ui1_filtered, corr_idxs_i1i2_filtered, two_view_reports_filtered
