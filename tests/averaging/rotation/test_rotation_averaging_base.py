"""Tests for rotation averaging base class.

Authors: Ayush Baid
"""

import pickle
import unittest

import dask
import numpy as np
from gtsam import Rot3

import utils.geometry_comparisons as geometry_comparisons
from averaging.rotation.dummy_rotation_averaging import DummyRotationAveraging


class TestRotationAveragingBase(unittest.TestCase):
    """Main tests for rotation averaging base class."""

    def setUp(self):
        super(TestRotationAveragingBase, self).setUp()

        self.obj = DummyRotationAveraging()

    def test_computation_graph(self):
        """Test the dask computation graph execution using a valid collection
        of relative poses."""

        num_poses = 3

        i2Ri1_dict = {
            (0, 1): Rot3.RzRyRx(0, np.deg2rad(30), 0),
            (1, 2): Rot3.RzRyRx(0, 0, np.deg2rad(20)),
        }

        i2Ri1_graph = dask.delayed(i2Ri1_dict)

        # use the GTSAM API directly (without dask) for rotation averaging
        expected_wRi_list = self.obj.run(num_poses, i2Ri1_dict)

        # use dask's computation graph
        computation_graph = self.obj.create_computation_graph(
            num_poses,
            i2Ri1_graph
        )

        with dask.config.set(scheduler='single-threaded'):
            wRi_list = dask.compute(computation_graph)[0]

        # compare the two results
        self.assertTrue(geometry_comparisons.compare_rotations(
            wRi_list, expected_wRi_list))

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


if __name__ == '__main__':
    unittest.main()
