"""Tests for rotation averaging base class.

Authors: Ayush Baid
"""

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
        """Test the dask computation graph execution."""

        num_poses = 5

        # pose indices for relative rotation input
        pose_pairs_ij = [
            (0, 1),
            (0, 2),
            (1, 3),
            (3, 4),
            (2, 4)
        ]

        relative_iRj = dict()

        # generate random relative rotations
        for i, j in pose_pairs_ij:
            random_vector = np.random.rand(3)*2*np.pi
            relative_iRj[(i, j)] = Rot3.Rodrigues(
                random_vector[0], random_vector[1], random_vector[2])

        # use the normal API for rotation averaging
        normal_result = self.obj.run(num_poses, relative_iRj)

        # use dask's computation graph
        computation_graph = self.obj.create_computation_graph(
            num_poses,
            dask.delayed(relative_iRj)
        )

        with dask.config.set(scheduler='single-threaded'):
            results = dask.compute(computation_graph)[0]

        # compare the two entries
        for idx in range(num_poses):
            self.assertTrue(normal_result[idx].equals(results[idx], 1e-5))
