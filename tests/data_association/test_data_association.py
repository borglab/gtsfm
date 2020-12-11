"""Unit test for the DataAssociation class.

Authors: Sushmita Warrier
"""
import unittest

import dask
import numpy as np
from common.keypoints import Keypoints
from gtsam import (
    Cal3Bundler,
    PinholeCameraCal3Bundler,
    Point3,
    Pose3,
    Pose3Vector,
    Rot3,
    SfmData
)
from gtsam.utils.test_case import GtsamTestCase

from data_association.dummy_da import DummyDataAssociation


class TestDataAssociation(GtsamTestCase):
    """Unit tests for data association module."""

    def setUp(self):
        """Set up the data association object and test data."""
        super().setUp()

        self.obj = DummyDataAssociation(0.5, 2)

        # set up ground truth data for comparison

        self.dummy_corr_idxs_dict = {
            (0, 1): np.array([[0, 2]]),
            (1, 2): np.array([[2, 3],
                              [4, 5],
                              [7, 9]]),
            (0, 2): np.array([[1, 8]])}
        self.keypoints_list = [
            Keypoints(coordinates=np.array([
                [12, 16],
                [13, 18],
                [0, 10]])),
            Keypoints(coordinates=np.array([
                [8, 2],
                [16, 14],
                [22, 23],
                [1, 6],
                [50, 50],
                [16, 12],
                [82, 121],
                [39, 60]])),
            Keypoints(coordinates=np.array([
                [1, 1],
                [8, 13],
                [40, 6],
                [82, 21],
                [1, 6],
                [12, 18],
                [15, 14],
                [25, 28],
                [7, 10],
                [14, 17]]))
        ]

        # Generate two poses for use in triangulation tests
        # Looking along X-axis, 1 meter above ground plane (x-y)
        upright = Rot3.Ypr(-np.pi / 2, 0.0, -np.pi / 2)
        pose1 = Pose3(upright, Point3(0, 0, 1))

        # create second camera 1 meter to the right of first camera
        pose2 = pose1.compose(Pose3(Rot3(), Point3(1, 0, 0)))

        self.poses = Pose3Vector()
        self.poses.append(pose1)
        self.poses.append(pose2)

        # landmark ~5 meters infront of camera
        self.expected_landmark = Point3(5, 0.5, 1.2)

    def test_create_computation_graph(self):
        """Test the dask computation graph."""
        sharedCal = Cal3Bundler(1500, 0, 0, 640, 480)
        cameras = {
            i: PinholeCameraCal3Bundler(x, sharedCal)
            for (i, x) in enumerate(self.poses)
        }

        camera_graph = dask.delayed(cameras)

        corr_idxs_graph = {k: dask.delayed(v) for (
            k, v) in self.dummy_corr_idxs_dict.items()}

        keypoints_graph = [dask.delayed(x) for x in self.keypoints_list]

        da_graph = self.obj.create_computation_graph(
            camera_graph, corr_idxs_graph, keypoints_graph)

        with dask.config.set(scheduler='single-threaded'):
            dask_result = dask.compute(da_graph)[0]

        self.assertIsInstance(dask_result, SfmData)


if __name__ == "__main__":
    unittest.main()
