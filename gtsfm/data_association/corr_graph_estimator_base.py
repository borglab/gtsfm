"""Implements a base class for CorrespondenceGraph estimation.

Authors: Travis Driver
"""

import abc

import dask
from dask.delayed import Delayed

from gtsfm.data_association.correspondence_graph import CorrespondenceGraph


class CorrespondenceGraphEstimatorBase(metaclass=abc.ABCMeta):
    """"""

    @abc.abstractmethod
    def run(self) -> CorrespondenceGraph:
        """Compute CorrespondenceGraph."""

    def create_computation_graph(self) -> Delayed:
        """Create Dask task graph for correspondences."""
        return dask.delayed(self.run)()
