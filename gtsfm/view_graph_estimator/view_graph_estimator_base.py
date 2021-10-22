"""Implements a base class for ViewGraph estimation.

Estimating the ViewGraph can be done trivially by adding all the two-view estimates into a ViewGraph data structure.
The purpose of this class, however, is to define an API for more sophistacated methods for estimating a ViewGraph 
that include filtering or optimizing the two-view estimates.

Authors: Akshay Krishnan
"""
import abc
from typing import Dict, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Rot3, Unit3

from gtsfm.view_graph_estimator.view_graph import ViewGraph
from gtsfm.evaluation.metrics import GtsfmMetricsGroup


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
        K: Dict[int, Cal3Bundler],
        correspondeces_i1_i2: Dict[Tuple[int, int], np.ndarray],
        i2Ei1: Dict[Tuple[int, int], np.ndarray] = None,
        i2Fi1: Dict[Tuple[int, int], np.ndarray] = None,
    ) -> Tuple[ViewGraph, GtsfmMetricsGroup]:
        """Run the ViewGraph estimation.

        Args:
            i2Ri1: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            K: Dict from camera idx to its intrinsic parameters (Cal3Bundler)
            correspondeces_i1_i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            i2Ei1: Dict from (i1, i2) to essential matrix between them (optional).
            i2Fi1: Dict from (i1, i2) to Fundamental matrix between them (optional).

        Returns:
            Tuple of a ViewGraph, metrics for ViewGraph estimation.
        """

    def create_computation_graph(
        self,
        i2Ri1: Delayed,
        i2Ui1: Delayed,
        K: Delayed,
        correspondeces_i1_i2: Delayed,
        i2Ei1: Delayed = None,
        i2Fi1: Delayed = None,
    ) -> Delayed:
        """Create the computation graph for ViewGraph estimation..

        Args:
            i2Ri1: Dict from (i1, i2) to relative rotation of i1 with respect to i2 (wrapped as Delayed).
            i2Ui1: Dict from (i1, i2) to relative translation direction of i1 with respect to i2 (wrapped as Delayed).
            K: Dict from camera idx to its intrinsic parameters (Cal3Bundler) (wrapped as Delayed).
            correspondeces_i1_i2: Dict from (i1, i2) to correspondences from i1 to i2  (wrapped as Delayed).
            i2Ei1: Dict from (i1, i2) to essential matrix between them (optional)  (wrapped as Delayed).
            i2Fi1: Dict from (i1, i2) to Fundamental matrix between them (optional)  (wrapped as Delayed).

        Returns:
            global rotations wrapped using dask.delayed.
        """

        return dask.delayed(self.run, nout=2)(i2Ri1, i2Ui1, K, correspondeces_i1_i2, i2Ei1, i2Fi1)
