"""Tests for the PoseSlam module.

Authors: Akshay Krishnan, Frank Dellaert
"""
import unittest
from typing import Dict, List, Optional, Tuple

import numpy as np
from dask.delayed import Delayed
from gtsam import Point3, Pose3, Rot3
from gtsam.utils.test_case import GtsamTestCase

from gtsfm.common.pose_prior import PosePrior, PosePriorType
from gtsfm.pose_slam.pose_slam import PoseSlam


class TestPoseSlam(GtsamTestCase):
    """Test class for 1DSFM rotation averaging."""

    def setUp(self) -> None:
        super().setUp()

        self.slam: PoseSlam = PoseSlam()

    def test_run_pose_slam(self) -> None:
        """Check that a simple pose graph works."""
        sigma = np.deg2rad(5)
        cov = np.diag([sigma, sigma, sigma, 0.1, 0.1, 0.1])
        relative_pose_priors = {
            (0, 1): PosePrior(Pose3(Rot3(), Point3(1, 0, 0)), cov, type=PosePriorType.SOFT_CONSTRAINT),
            (0, 2): PosePrior(Pose3(Rot3(), Point3(2, 0, 0)), cov, type=PosePriorType.SOFT_CONSTRAINT),
            (1, 2): PosePrior(Pose3(Rot3(), Point3(1, 0, 0)), cov, type=PosePriorType.SOFT_CONSTRAINT),
        }

        gt_wTi_list = [None, None, None]
        poses, _ = self.slam.run_pose_slam(3, relative_pose_priors=relative_pose_priors, gt_wTi_list=gt_wTi_list)
        self.assertEqual(len(poses), 3)
        for pose in poses:
            self.assertIsInstance(pose, Pose3)
        self.gtsamAssertEquals(poses[0], Pose3())
        self.gtsamAssertEquals(poses[1], Pose3(Rot3(), Point3(1, 0, 0)))
        self.gtsamAssertEquals(poses[2], Pose3(Rot3(), Point3(2, 0, 0)))

    def test_create_computation_graph(self) -> None:
        """Check that the interface works."""
        relative_pose_priors: Dict[Tuple[int, int], PosePrior] = {}
        gt_wTi_list: Optional[List[Optional[Pose3]]] = [None, None, None]
        delayed_poses, metrics = self.slam.create_computation_graph(
            3, relative_pose_priors=relative_pose_priors, gt_wTi_list=gt_wTi_list
        )
        self.assertIsInstance(delayed_poses, Delayed)
        self.assertIsInstance(metrics, Delayed)


if __name__ == "__main__":
    unittest.main()
