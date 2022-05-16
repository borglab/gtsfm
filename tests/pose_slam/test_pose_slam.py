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
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtParams,
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


def read_fastlio_result(poses_txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
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
    return poses, timestamps


def pose_of_vector(pose_vector):
    """Convert from vector to Pose3"""
    pose_rotation = Rot3(pose_vector[-1], pose_vector[3], pose_vector[4], pose_vector[5])
    pose_translation = Point3(pose_vector[:3])
    return Pose3(pose_rotation, pose_translation)


def get_relative_transform(pose1_vec, pose2_vec):
    """
    Computes the relative transform given two pose vectors read from the fastlio txt file.
    Args:
        pose1_vec: [x,y,z,q1,q2,q3,q4(w)]
        pose2_vec: [x,y,z,q1,q2,q3,q4(w)]
    Returns:
        T2_1: relative transform from pose1 to pose2
    """
    pose1 = pose_of_vector(pose1_vec)
    pose2 = pose_of_vector(pose2_vec)
    return pose2.between(pose1)


def angle(R1, R2):
    """Calculate angle between two rotations, in degrees."""
    return np.degrees(np.linalg.norm(R1.logmap(R2)))


def difference(P1, P2):
    """Calculate the translation and angle differences of two poses.
    P1, P2: Pose3
    Return:
        distance: translation difference
        angle: angular difference
    """
    # TODO(frank): clean this up
    t1 = P1.translation()
    t2 = P2.translation()
    R1 = P1.rotation()
    R2 = P2.rotation()
    R1_2 = R1.compose(R2.inverse())
    t1_ = R1_2.rotate(t2)
    # t1_2 = t1 - R1_2*t2
    distance = np.linalg.norm(t1 - t1_)
    angle_ = angle(R1, R2)
    return distance, angle_


def create_initial_estimate(poses):
    """Create initial estimate"""
    initial_estimate = Values()
    for i, pose_vector in enumerate(poses):
        initial_estimate.insert(i, pose_of_vector(pose_vector))
    return initial_estimate


def generate_pose_graph(constraints, poses, initial_estimate, add_backbone=True):
    """Generate pose graph
    To create the actual factor for the factor graph and optimize the result.
    This process includes:
        1. Initialize the factor graph
        2. Create the pose nodes
        3. Create the between factors
        4. Optimize the graph
        5. Save the result
    Args:
        A list of factors
    """
    # Initialize the factor graph
    graph = NonlinearFactorGraph()

    # Add the prior factor to the initial pose.
    prior_pose_factor = PriorFactorPose3(0, pose_of_vector(poses[0]), noiseModel.Isotropic.Sigma(6, 5e-2))
    graph.add(prior_pose_factor)

    # Create loose odometry factors
    if add_backbone:
        backbone_noise = noiseModel.Diagonal.Sigmas(
            np.array([1, 1, 1, np.deg2rad(30.0), np.deg2rad(30.0), np.deg2rad(30.0)])
        )
        for i in range(len(poses) - 1):
            a, b = i, i + 1
            a_vec, b_vec = poses[a], poses[b]
            bTa = get_relative_transform(a_vec, b_vec)
            factor_aTb = BetweenFactorPose3(i, i + 1, bTa.inverse(), backbone_noise)
            graph.add(factor_aTb)

    # Create pose constraints using Bayesian ICP
    for constraint in constraints:
        a, b = constraint.a, constraint.b
        aTb, cov = constraint.aTb, constraint.cov
        a_vec, b_vec = poses[a], poses[b]
        bTa = get_relative_transform(a_vec, b_vec)  # fastlio
        trans_diff, rot_diff = difference(aTb, bTa.inverse())
        inlier = (trans_diff <= 0.04) and (rot_diff <= 5)
        if not inlier:
            continue
        try:
            info = np.linalg.inv(cov)
            if np.isnan(info).any():
                continue
            else:
                measurement_noise = noiseModel.Gaussian.Information(info)
                factor_aTb = BetweenFactorPose3(a, b, aTb, measurement_noise)
                error = factor_aTb.error(initial_estimate)
                if np.isnan(error):
                    continue
                if error > 1000:
                    continue
                else:
                    graph.add(factor_aTb)
        except np.linalg.LinAlgError:
            continue

    # Output the resulting poses and pose constraints
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
        ws = Path(EXP07_PATH)
        poses_txt_path = ws / "fastlio_odom.txt"
        constraint_txt_path = ws / "constraints.txt"
        poses, _ = read_fastlio_result(str(poses_txt_path))
        constraints = Constraint.read(str(constraint_txt_path))
        self.assertEqual(len(poses), 1319)
        self.assertEqual(len(constraints), 10328)

        # graph, initial_estimate = self.slam.generate_pose_graph(constraints, poses)
        initial_estimate = create_initial_estimate(poses)
        graph = generate_pose_graph(constraints, poses, initial_estimate)
        self.assertEqual(graph.size(), 10983)
        self.assertAlmostEqual(graph.error(initial_estimate), 35111.87458699957)

        params = LevenbergMarquardtParams()
        optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        final_error = graph.error(result)
        self.assertAlmostEqual(final_error, 10288.499767263707)


if __name__ == "__main__":
    unittest.main()
