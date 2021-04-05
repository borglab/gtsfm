"""Tests for rotation averaging base class.

Authors: Ayush Baid
"""

import pickle
import unittest
from typing import Dict, List, Optional, Tuple

import dask
import numpy as np
from gtsam import Rot3

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import tests.data.sample_poses as sample_poses
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase

ROTATION_ANGLE_ERROR_THRESHOLD_DEG = 2


class TestRotationAveragingBase(unittest.TestCase):
    """Main tests for rotation averaging base class."""

    def setUp(self):
        super(TestRotationAveragingBase, self).setUp()

        self.obj = DummyRotationAveraging()

    def __execute_test(self, i2Ri1_input: Dict[Tuple[int, int], Rot3], wRi_expected: List[Rot3]) -> None:
        """Helper function to run the averagaing and assert w/ expected.

        Args:
            i2Ri1_input: relative rotations, which are input to the algorithm.
            wRi_expected: expected global rotations.
        """
        if isinstance(self.obj, DummyRotationAveraging):
            self.skipTest("Test case invalid for dummy class")

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


class DummyRotationAveraging(RotationAveragingBase):
    """Assigns random rotation matrices to each pose."""

    def run(self, num_images: int, i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]]) -> List[Optional[Rot3]]:
        """Run the rotation averaging.

        Args:
            num_images: number of poses.
            i2Ri1_dict: relative rotations as dictionary (i1, i2): i2Ri1.

        Returns:
            Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
                `num_images`. The list may contain `None` where the global rotation could not be computed (either
                underconstrained system or ill-constrained system).
        """
        if len(i2Ri1_dict) == 0:
            return [None] * num_images

        # create the random seed using relative rotations
        seed_rotation = next(iter(i2Ri1_dict.values()))
        np.random.seed(int(1000 * seed_rotation.xyz()[0]) % (2 ^ 32))

        # TODO: do not assign values where we do not have any edge

        # generate dummy rotations
        wRi_list = []
        for _ in range(num_images):
            random_vector = np.random.rand(3) * 2 * np.pi
            wRi_list.append(Rot3.Rodrigues(random_vector[0], random_vector[1], random_vector[2]))

        return wRi_list


if __name__ == "__main__":
    unittest.main()
