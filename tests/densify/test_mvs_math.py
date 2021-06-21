"""Unit tests for math utilities for MVS methods

Authors: Ren Liu
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

import gtsfm.densify.mvs_math as mvs_math


class TestMVSMath(unittest.TestCase):
    """Unit tests for math utilities for MVS methods."""

    def test_piecewise_gaussian_below_threshold(self) -> None:
        """Unit test for the case that the angle between two coordinates is below the threshold,
        where sigma_1 is used to calculate the score"""

        xPa = np.array([15.0, 7.5, 0.0])
        rot = R.from_euler("z", 4, degrees=True)
        xPb = rot.apply(xPa)

        score = mvs_math.piecewise_gaussian(xPa=xPa, xPb=xPb, theta_0=5, sigma_1=1, sigma_2=10)

        self.assertAlmostEqual(score, np.exp(-(1.0 ** 2) / (2 * 1.0 ** 2)))

    def test_piecewise_gaussian_above_threshold(self) -> None:
        """Unit test for the case that the angle between two coordinates is above the threshold,
        where sigma_2 is used to calculate the score"""

        xPa = np.array([0.0, 12.0, 15.0])
        rot = R.from_euler("x", 10, degrees=True)
        xPb = rot.apply(xPa)

        score = mvs_math.piecewise_gaussian(xPa=xPa, xPb=xPb, theta_0=5, sigma_1=1, sigma_2=10)

        self.assertAlmostEqual(score, np.exp(-(5.0 ** 2) / (2 * 10.0 ** 2)))


if __name__ == "__main__":
    unittest.main()
