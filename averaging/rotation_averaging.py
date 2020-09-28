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

    def __init__(self, pose_num: int, measurements: dask.delayed(dict((int, int), (np.array, np.array, np.array)))):
        """
        Initialize shonan from measurements.

        Args:
            measurements (ShonanAveraging3.Measurements): the measurements
            params (ShonanAveraging3.Parameters): the parameters for shonan
        """
        shonan_measurements = []
        for key, item in measurements.items():
            # TODO(Jing): how to get the noise model
            noise_model = item[2]
            # TODO(Jing): make sure the correctness of converting key and rotation into binary_measurement_rot3 type
            binary_measurements_rot3 = gtsam.BinaryMeasurementRot3(
                key[0], key[1], item[0], gtsam.noiseModel.Gaussian.Covariance(noise_model[:3, :3], True))
            shonan_measurements.append(binary_measurements_rot3)

        self.params = ShonanAveraging3.ShonanAveragingParameters()
        self.pose_num = pose_num
        self.shonan_measurements = shonan_measurements
        self.shonan = ShonanAveraging3(shonan_measurements, self.params)
        self._pMin = 5
        self._pMax = 30

    def run(self) -> dask.delayed(dict((int, int)), (np.array))):
        """
        Really run Shonan

        Arguments:
            initial (Values): the initial estimation in Rot3 type
            solver (str): the type of the solver

        Returns:
            Values: the result estimation in Rot3 type
            np.ndarray: the descriptors for the input features
        """
        initial = gtsam.Values()
        for i in range(self.pose_num):
            initial.insert(i, self.random_Rot3())
        result = self.shonan.run(initial, self._pMin, self._pMax)
        #TODO(Jing): convert result into dask delayed type
        return dask.delayed(result)

    def random_Rot3(self):
        """
        Create random Values in ROt3 type

        Returns:
            Value: one estimation in Rot3 type
        """
        u = np.random.randn(3)
        theta = np.random.uniform(-np.pi, np.pi)
        return gtsam.Rot3.Rodrigues(u*(theta/np.linalg.norm(u)))
