"""Tests for rotation averaging base class.

Authors: Ayush Baid
"""

import unittest

import dask
import numpy as np
from averaging.rotation.dummy_rotation_averaging import DummyRotationAveraging
from gtsam import Rot3


class TestRotationAveragingBase(unittest.TestCase):
    """Main tests for rotation averaging base class."""

    def setUp(self):
        super(TestRotationAveragingBase, self).setUp()

        self.obj = DummyRotationAveraging()

    def test_computation_graph(self):
        """Test the dask computation graph execution."""

        num_poses = 5

        # pose indices for relative rotation input
        pose_pairs = [
            (0, 1),
            (0, 2),
            (1, 3),
            (3, 4),
            (2, 4)
        ]

        relative_rotations = dict()

        # generate random relative rotations
        for pair_key in pose_pairs:
            random_vector = np.random.rand(3)*2*np.pi
            relative_rotations[pair_key] = Rot3.Rodrigues(
                random_vector[0], random_vector[1], random_vector[2])

        # use the normal API for rotation averaging
        normal_result = self.obj.run(num_poses, relative_rotations)

        # use dask's computation graph
        computation_graph = self.obj.create_computation_graph(
            num_poses,
            dask.delayed(relative_rotations)
        )

        with dask.config.set(scheduler='single-threaded'):
            results = dask.compute(computation_graph)[0]

        # compare the two entries
        for idx in range(num_poses):
            self.assertTrue(normal_result[idx].equals(results[idx], 1e-5))
