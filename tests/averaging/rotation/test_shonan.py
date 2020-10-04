"""Tests for Shonan rotation averaging.

Authors: Ayush Baid
"""


import numpy as np

from gtsam import Rot3

import tests.averaging.rotation.test_rotation_averaging_base as test_rotation_averaging_base

from averaging.rotation.shonan import ShonanRotationAveraging


class TestShonanRotationAveraging(
        test_rotation_averaging_base.TestRotationAveragingBase):
    """Test class for Shonan rotation averaging."""

    def setUp(self):
        super().setUp()

        self.obj = ShonanRotationAveraging()

    def test_simple(self):
        """Test a simple case with just three relative rotations."""

        iRj_dict = {
            (0, 1): Rot3.RzRyRx(0, 30*np.pi/180, 0),
            (1, 2): Rot3.RzRyRx(0, 0, 20*np.pi/180),
        }

        expected_0Ri = [
            Rot3.RzRyRx(0, 0, 0),
            Rot3.RzRyRx(0, 30*np.pi/180, 0),
            iRj_dict[(0, 1)].compose(iRj_dict[(1, 2)])
        ]

        computed_wRi = self.obj.run(3, iRj_dict)

        self.assertTrue(expected_0Ri[1].equals(
            computed_wRi[0].between(computed_wRi[1]), 1e-5))
        self.assertTrue(expected_0Ri[2].equals(
            computed_wRi[0].between(computed_wRi[2]), 1e-5))
