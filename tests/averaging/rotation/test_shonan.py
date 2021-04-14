"""Tests for Shonan rotation averaging.

Authors: Ayush Baid
"""
import pickle
import unittest
from typing import Dict, List, Tuple

import dask
import numpy as np
from gtsam import Rot3

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import tests.data.sample_poses as sample_poses
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging

ROTATION_ANGLE_ERROR_THRESHOLD_DEG = 2


class TestShonanRotationAveraging(unittest.TestCase):
    """Test class for Shonan rotation averaging.

    All unit test functions defined in TestRotationAveragingBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.obj: RotationAveragingBase = ShonanRotationAveraging()

    def __execute_test(self, i2Ri1_input: Dict[Tuple[int, int], Rot3], wRi_expected: List[Rot3]) -> None:
        """Helper function to run the averagaing and assert w/ expected.

        Args:
            i2Ri1_input: relative rotations, which are input to the algorithm.
            wRi_expected: expected global rotations.
        """
        wRi_computed = self.obj.run(len(wRi_expected), i2Ri1_input)
        self.assertTrue(
            geometry_comparisons.compare_rotations(wRi_computed, wRi_expected, ROTATION_ANGLE_ERROR_THRESHOLD_DEG)
        )

    def test_circle_two_edges(self):
        """Test for 4 poses in a circle, with a pose connected to its immediate neighborhood."""
        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_TWO_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ri1_dict, wRi_expected)

    def test_circle_all_edges(self):
        """Test for 4 poses in a circle, with a pose connected all others."""
        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.CIRCLE_ALL_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_ALL_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ri1_dict, wRi_expected)

    def test_line_large_edges(self):
        """Test for 3 poses in a line, with large translations between them."""
        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.LINE_LARGE_EDGES_GLOBAL_POSES, sample_poses.LINE_LARGE_EDGES_RELATIVE_POSES
        )
        self.__execute_test(i2Ri1_dict, wRi_expected)

    def test_panorama(self):
        """Test for 3 poses in a panorama configuration (large rotations at the same location)"""
        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.PANORAMA_GLOBAL_POSES, sample_poses.PANORAMA_RELATIVE_POSES
        )
        self.__execute_test(i2Ri1_dict, wRi_expected)

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

        self.__execute_test(i2Ri1_dict, expected_wRi_list)

    def test_computation_graph(self):
        """Test the dask computation graph execution using a valid collection of relative poses."""

        num_poses = 3

        i2Ri1_dict = {
            (0, 1): Rot3.RzRyRx(0, np.deg2rad(30), 0),
            (1, 2): Rot3.RzRyRx(0, 0, np.deg2rad(20)),
        }

        i2Ri1_graph = dask.delayed(i2Ri1_dict)

        # use the GTSAM API directly (without dask) for rotation averaging
        expected_wRi_list = self.obj.run(num_poses, i2Ri1_dict)

        # use dask's computation graph
        computation_graph = self.obj.create_computation_graph(num_poses, i2Ri1_graph)

        with dask.config.set(scheduler="single-threaded"):
            wRi_list = dask.compute(computation_graph)[0]

        # compare the two results
        self.assertTrue(
            geometry_comparisons.compare_rotations(wRi_list, expected_wRi_list, ROTATION_ANGLE_ERROR_THRESHOLD_DEG)
        )

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


if __name__ == "__main__":
    unittest.main()
