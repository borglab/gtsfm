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

        i1_R_i2_dict = {
            (0, 1): Rot3.RzRyRx(0, 30*np.pi/180, 0),
            (1, 2): Rot3.RzRyRx(0, 0, 20*np.pi/180),
        }

        expected_w_R_i = [
            Rot3.RzRyRx(0, 0, 0),
            Rot3.RzRyRx(0, 30*np.pi/180, 0),
            i1_R_i2_dict[(0, 1)].compose(i1_R_i2_dict[(1, 2)])
        ]

        computed_w_R_i = self.obj.run(3, i1_R_i2_dict)

        self.assertTrue(expected_w_R_i[1].equals(
            computed_w_R_i[0].between(computed_w_R_i[1]), 1e-5))
        self.assertTrue(expected_w_R_i[2].equals(
            computed_w_R_i[0].between(computed_w_R_i[2]), 1e-5))


if __name__ == '__main__':
    unittest.main()
