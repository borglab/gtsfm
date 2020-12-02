"""Tests for rotation averaging base class.

Authors: Ayush Baid
"""

import pickle
import unittest

import dask
import numpy as np
from gtsam import Rot3

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
        expected_result = self.obj.run(num_poses, i2Ri1_dict)

        # use dask's computation graph
        computation_graph = self.obj.create_computation_graph(
            num_poses,
            i2Ri1_graph
        )

        with dask.config.set(scheduler='single-threaded'):
            dask_result = dask.compute(computation_graph)[0]

        # compare the two entries
        for i in range(1, num_poses):
            expected_0Ri = expected_result[0].between(expected_result[i])
            computed_0Ri = dask_result[0].between(dask_result[i])
            self.assertTrue(expected_0Ri.equals(computed_0Ri, 1e-5))

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


if __name__ == '__main__':
    unittest.main()
