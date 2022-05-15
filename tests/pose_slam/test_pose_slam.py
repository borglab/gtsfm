"""Tests for 1DSFM translation averaging.

Authors: Ayush Baid
"""
import unittest
from typing import Dict, List, Optional, Tuple

import dask
import gtsam
import numpy as np
from dask.delayed import Delayed
from gtsam import Point3, Pose3, Rot3, Unit3
from gtsam.utils.test_case import GtsamTestCase
from gtsfm.common.pose_prior import PosePrior, PosePriorType
from gtsfm.evaluation.metrics import GtsfmMetricsGroup

import gtsfm.utils.geometry_comparisons as geometry_comparisons
# from gtsfm.pose_slam.pose_slam import PoseSlam

PRIOR_NOISE_SIGMAS = [0.001, 0.001, 0.001, 0.1, 0.1, 0.1]
POSE_PRIOR_NOISE  = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_NOISE_SIGMAS)

class PoseSlam:
    def run_pose_slam(
        self,
        num_images: int,
        relative_pose_priors: Dict[Tuple[int, int], PosePrior] = {},
        gt_wTi_list: Optional[List[Optional[Pose3]]] = None,
    ) -> Tuple[List[Optional[Point3]], Optional[GtsfmMetricsGroup]]:
        """Run the translation averaging.

        Args:
            num_images: number of camera poses.
            relative_pose_priors: priors on the pose between camera pairs as (i1, i2): i1Ti2.
            gt_wTi_list: List of ground truth poses (wTi) for computing metrics.

        Returns:
            List of Optional[Pose3], 
            GtsfmMetricsGroup of 1DSfM metrics.
        """
        pose_init_graph = gtsam.NonlinearFactorGraph()

        for (i1, i2), i1Ti2_prior in relative_pose_priors.items():
            pose_init_graph.push_back(
                gtsam.BetweenFactorPose3(
                    i1,
                    i2,
                    i1Ti2_prior.value,
                    gtsam.noiseModel.Gaussian.Covariance(i1Ti2_prior.covariance),
                )
            )
        pose_init_graph.push_back(
          gtsam.PriorFactorPose3(0, Pose3(), POSE_PRIOR_NOISE)
        )

        values = gtsam.InitializePose3.initialize(pose_init_graph)
        poses = [values.atPose3(i) if values.exists(i) else None for i in range(num_images)]

        return poses, None

    def create_computation_graph(
        self,
        num_images: int,
        relative_pose_priors: Dict[Tuple[int, int], PosePrior] = {},
        gt_wTi_list: Optional[List[Optional[Pose3]]] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing translation averaging.

        Args:
            num_images: number of camera poses.
            relative_pose_priors: priors on the pose between camera pairs (not delayed).
            gt_wTi_list: List of ground truth poses (wTi) for computing metrics.

        Returns:
            List of Optional[Pose3], wrapped as Delayed
            GtsfmMetricsGroup of 1DSfM metrics, wrapped as Delayed
        """
        return dask.delayed(self.run_pose_slam, nout=2)(
            num_images, relative_pose_priors, gt_wTi_list
        )

class TestPoseSlam(GtsamTestCase):
    """Test class for 1DSFM rotation averaging."""

    def setUp(self):
        super().setUp()

        self.slam: PoseSlam = PoseSlam()

    def test_create_computation_graph(self):
        """Check that the interface works."""
        relative_pose_priors = {}
        gt_wTi_list = [None, None, None]
        delayed_poses, metrics = self.slam.create_computation_graph(
            3, relative_pose_priors=relative_pose_priors, gt_wTi_list=gt_wTi_list
        )
        self.assertIsInstance(delayed_poses, Delayed)
        self.assertIsInstance(metrics, Delayed)

    def test_run_pose_slam(self):
        """Check that a simple pose graph works."""
        sigma = np.deg2rad(5)
        cov = np.diag([sigma, sigma, sigma, 0.1, 0.1, 0.1])
        relative_pose_priors = {
          (0, 1): PosePrior(Pose3(Rot3(), Point3(1, 0, 0)), cov, type=PosePriorType.SOFT_CONSTRAINT),
          (0, 2): PosePrior(Pose3(Rot3(), Point3(2, 0, 0)), cov, type=PosePriorType.SOFT_CONSTRAINT),
          (1, 2): PosePrior(Pose3(Rot3(), Point3(1, 0, 0)), cov, type=PosePriorType.SOFT_CONSTRAINT),
        }

        gt_wTi_list = [None, None, None]
        poses, metrics = self.slam.run_pose_slam(
            3, relative_pose_priors=relative_pose_priors, gt_wTi_list=gt_wTi_list
        )
        self.assertEqual(len(poses), 3)
        for pose in poses:
            self.assertIsInstance(pose, Pose3)
        self.gtsamAssertEquals(poses[0], Pose3())
        self.gtsamAssertEquals(poses[1], Pose3(Rot3(), Point3(1, 0, 0)))
        self.gtsamAssertEquals(poses[2], Pose3(Rot3(), Point3(2, 0, 0)))
        


    def test_create_computation_graph(self):
        """Check that the interface works."""
        relative_pose_priors = {}
        gt_wTi_list = [None, None, None]
        delayed_poses, metrics = self.slam.create_computation_graph(
            3, relative_pose_priors=relative_pose_priors, gt_wTi_list=gt_wTi_list
        )
        self.assertIsInstance(delayed_poses, Delayed)
        self.assertIsInstance(metrics, Delayed)


if __name__ == "__main__":
    unittest.main()
