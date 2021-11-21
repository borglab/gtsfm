"""Implements a base class for ViewGraph estimation.

Estimating the ViewGraph can be done trivially by adding all the two-view estimates into a ViewGraph data structure.
The purpose of this class, however, is to define an API for more sophisticated methods for estimating a ViewGraph 
that include filtering or optimizing the two-view estimates.

Authors: Akshay Krishnan, Ayush Baid
"""
import abc
from typing import Dict, Optional, List, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.view_graph import ViewGraph

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
    ) -> ViewGraph:
        """Run the ViewGraph estimation.

        Args:
            i2Ri1: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            calibrations: list of calibrations for each image.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            keypoints: keypoints for each images.

        Returns:
            ViewGraph object.
        """

    def compute_metrics(
        self, view_graph: ViewGraph, gt_cameras: Optional[List[PinholeCameraCal3Bundler]]
    ) -> GtsfmMetricsGroup:
        """Compute the metrics for the view graph estimation.

        Args:
            view_graph: view graph computed by the `run` method.
            gt_cameras: ground truth cameras to compute the metrics against.

        Returns:
            Metrics for the computed view graph.
        """
        if gt_cameras is None:
            return GtsfmMetricsGroup(name=METRIC_GROUP, metrics=[])

        rotation_errors_deg = []
        translation_errors_deg = []

        for i1, i2 in view_graph.get_pair_indices():
            i2Ti1_expected = gt_cameras[i2].pose().between(gt_cameras[i1].pose())

            R_error_deg = comp_utils.compute_relative_rotation_angle(
                view_graph.i2Ri1[(i1, i2)], i2Ti1_expected.rotation()
            )
            U_error_deg = comp_utils.compute_relative_unit_translation_angle(
                view_graph.i2Ui1[(i1, i2)], Unit3(i2Ti1_expected.translation())
            )

            rotation_errors_deg.append(R_error_deg)
            translation_errors_deg.append(U_error_deg)

        return GtsfmMetricsGroup(
            name=METRIC_GROUP,
            metrics=[
                GtsfmMetric(name="relative_rotation_errors", data=rotation_errors_deg),
                GtsfmMetric(name="relative_direction_errors", data=translation_errors_deg),
            ],
        )

    def create_computation_graph(
        self,
        i2Ri1: Delayed,
        i2Ui1: Delayed,
        calibrations: Delayed,
        corr_idxs_i1i2: Delayed,
        keypoints: Delayed,
        i2Ti1_gt: Delayed,
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for ViewGraph estimation and metric evaluation.

        The input arguments and the outputs of the functions are the same as the `run` method, but wrapped in Delayed.
        """
        view_graph = dask.delayed(self.run, nout=2)(i2Ri1, i2Ui1, calibrations, corr_idxs_i1i2, keypoints)
        metrics = dask.delayed(self.compute_metrics)(view_graph, i2Ti1_gt)

        return view_graph, metrics
