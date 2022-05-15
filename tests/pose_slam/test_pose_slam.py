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
from gtsfm.common.pose_prior import PosePrior
from gtsfm.evaluation.metrics import GtsfmMetricsGroup

import gtsfm.utils.geometry_comparisons as geometry_comparisons
# from gtsfm.pose_slam.pose_slam import PoseSlam


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

class TestPoseSlam(unittest.TestCase):
    """Test class for 1DSFM rotation averaging."""

    def setUp(self):
        super().setUp()

        self.slam: PoseSlam = PoseSlam()

    def test_all(self):
        """Check that the interface works."""
        relative_pose_priors = {}
        gt_wTi_list = [None, None, None]
        delayed_poses = self.slam.create_computation_graph(
            3, relative_pose_priors=relative_pose_priors, gt_wTi_list=gt_wTi_list
        )        



if __name__ == "__main__":
    unittest.main()
