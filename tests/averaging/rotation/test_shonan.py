"""Tests for Shonan rotation averaging.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import Rot3

import tests.averaging.rotation.test_rotation_averaging_base as \
    test_rotation_averaging_base
from averaging.rotation.shonan import ShonanRotationAveraging


class TestShonanRotationAveraging(
        test_rotation_averaging_base.TestRotationAveragingBase):
    """Test class for Shonan rotation averaging.

    All unit test functions defined in TestRotationAveragingBase are run
    automatically.
    """

    def setUp(self):
        super().setUp()

        self.obj = ShonanRotationAveraging()

    def test_simple(self):
        """Test a simple case with three relative rotations."""

        i2Ri1_dict = {
            (0, 1): Rot3.RzRyRx(0, 30*np.pi/180, 0),
            (1, 2): Rot3.RzRyRx(0, 0, 20*np.pi/180),
        }

        wRi_expected = [
            Rot3.RzRyRx(0, 0, 0),
            Rot3.RzRyRx(0, 30*np.pi/180, 0),
            i2Ri1_dict[(0, 1)].compose(i2Ri1_dict[(1, 2)])
        ]

        wRi_computed = self.obj.run(3, i2Ri1_dict)

        self.assertTrue(wRi_expected[1].equals(
            wRi_computed[0].between(wRi_computed[1]), 1e-5))
        self.assertTrue(wRi_expected[2].equals(
            wRi_computed[0].between(wRi_computed[2]), 1e-5))


if __name__ == '__main__':
    unittest.main()
