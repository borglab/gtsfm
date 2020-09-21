""" 
Class for handling rotation averaging with Shonan Averaging.

Authors: Jing Wu
"""

import dask
import numpy as np

import gtsam
from gtsam import (Rot3, Values)
from gtsam import ShonanAveraging3

class RotationAveraging():
    """
    Class for calling ShonanAveraging algorithm in gtsam
    """

    def __init__(self, measurements: ShonanAveraging3.Measurements, params: ShonanAveraging3.Parameters):
        """
        Initialize shonan from measurements and parameters.

        Args:
            measurements (ShonanAveraging3.Measurements): the measurements
            params (ShonanAveraging3.Parameters): the parameters for shonan
        """
        self.params = params
        self.shonan = ShonanAveraging3(measurements, params)
        self._pMin = 5
        self._pMax = 30

    def run(self, intial: dask.delayed(Values)) -> tuple(dask.delayed(Values), dask.delayed(np.array)):
        """
        Really run Shonan

        Arguments:
            initial (Values): the initial estimation in Rot3 type
            solver (str): the type of the solver

        Returns:
            Values: the result estimation in Rot3 type
            np.ndarray: the descriptors for the input features
        """
        return dask.delayed(self.shonan.run)(initial, self._pMin, self._pMax)
