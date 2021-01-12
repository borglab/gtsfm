"""Tests for Shonan rotation averaging.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import Rot3

import tests.averaging.rotation.test_rotation_averaging_base as test_rotation_averaging_base
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging


class TestShonanRotationAveraging(
    test_rotation_averaging_base.TestRotationAveragingBase
):
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
            (1, 0): Rot3.RzRyRx(0, np.deg2rad(30), 0),
            (2, 1): Rot3.RzRyRx(0, 0, np.deg2rad(20)),
        }

        expected_wRi_list = [
            Rot3.RzRyRx(0, 0, 0),
            Rot3.RzRyRx(0, np.deg2rad(30), 0),
            i2Ri1_dict[(1, 0)].compose(i2Ri1_dict[(2, 1)]),
        ]

        wRi_list = self.obj.run(3, i2Ri1_dict)

        self.assertTrue(
            expected_wRi_list[1].equals(wRi_list[0].between(wRi_list[1]), 1e-5)
        )
        self.assertTrue(
            expected_wRi_list[2].equals(wRi_list[0].between(wRi_list[2]), 1e-5)
        )


if __name__ == "__main__":
    unittest.main()
