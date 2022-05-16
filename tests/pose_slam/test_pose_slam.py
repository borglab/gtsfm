"""Tests for the PoseSlam module.

Authors: Akshay Krishnan, Frank Dellaert
"""
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dask.delayed import Delayed
from gtsam import (
    BetweenFactorPose3,
    InitializePose3,
    LevenbergMarquardtOptimizer,
    NonlinearFactorGraph,
    Point3,
    Pose3,
    PriorFactorPose3,
    Rot3,
    Values,
    noiseModel,
)
from gtsam.utils.test_case import GtsamTestCase

from gtsfm.common.constraint import Constraint
from gtsfm.common.pose_prior import PosePrior, PosePriorType
from gtsfm.pose_slam.pose_slam import PoseSlam

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
EXP07_PATH = DATA_ROOT_PATH / "exp07"


def pose_of_vector(pose_vector):
    """Convert from vector to Pose3"""
    pose_rotation = Rot3(pose_vector[-1], pose_vector[3], pose_vector[4], pose_vector[5])
    pose_translation = Point3(pose_vector[:3])
    return Pose3(pose_rotation, pose_translation)


def read_fastlio_result(poses_txt_path: str) -> Tuple[List[Pose3], np.ndarray]:
    """
    Read FastLIO result
    Args:
        1. The poses txt file path
        2. The covariance txt file path
    Returns:
        1. (poses, covariances)
    """
    # remove timestamp column from txt
    poses = np.loadtxt(poses_txt_path)[:, 1:]
    timestamps = np.loadtxt(poses_txt_path)[:, 0]
    return [pose_of_vector(pose) for pose in poses], timestamps


def create_initial_estimate(poses: List[Pose3]):
    """Create initial estimate"""
    initial_estimate = Values()
    for i, pose_vector in enumerate(poses):
        initial_estimate.insert(i, pose_vector)
    return initial_estimate


def generate_pose_graph(poses: List[Pose3], relative_pose_priors: Dict[Tuple[int, int], PosePrior]):
    """Generate pose graph."""
    graph = NonlinearFactorGraph()

    # Add the prior factor to the initial pose.
    prior_pose_factor = PriorFactorPose3(0, poses[0], noiseModel.Isotropic.Sigma(6, 5e-2))
    graph.add(prior_pose_factor)

    # Add pose priors
    for (a, b), pose_prior in relative_pose_priors.items():
        measurement_noise = noiseModel.Gaussian.Covariance(pose_prior.covariance)
        factor_aTb = BetweenFactorPose3(a, b, pose_prior.value, measurement_noise)
        graph.add(factor_aTb)

    return graph


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

    def test_filter_constraints(self) -> None:
        """Check that filtering and then doing pose slam on exp07 works as in notebook."""
        # Read constraints and poses from file.
        ws = Path(EXP07_PATH)
        poses_txt_path = ws / "fastlio_odom.txt"
        constraint_txt_path = ws / "constraints.txt"
        poses, _ = read_fastlio_result(str(poses_txt_path))
        constraints = Constraint.read(str(constraint_txt_path))
        self.assertEqual(len(poses), 1319)
        self.assertEqual(len(constraints), 10328)

        # Create graph.
        relative_pose_priors = self.slam.filtered_pose_priors(constraints, poses)
        graph = generate_pose_graph(poses, relative_pose_priors)
        self.assertEqual(graph.size(), 10062)

        # Check initial error.
        initial_estimate = create_initial_estimate(poses)
        self.assertAlmostEqual(graph.error(initial_estimate), 35111.87458699957)

        # Check that we can initialize with Pose3Initialize
        initial_estimate = InitializePose3.initialize(graph)
        self.assertAlmostEqual(graph.error(initial_estimate), 267965.28657262615)

        # Optimize and Check final error.
        optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate)
        result = optimizer.optimize()
        final_error = graph.error(result)
        self.assertAlmostEqual(final_error, 10288.082955348931)


if __name__ == "__main__":
    unittest.main()
