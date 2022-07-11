"""Unit tests for the hilti loader.

Note: currently running on the whole dataset.
"""
import unittest
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
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

from gtsfm.common.constraint import Constraint
from gtsfm.loader.hilti_loader import HiltiLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
TEST_DATASET_DIR_PATH = DATA_ROOT_PATH / "hilti_exp4_small"
EXP07_PATH = DATA_ROOT_PATH / "exp07"


def pose_of_vector(pose_vector):
    """Convert from vector to Pose3"""
    pose_rotation = Rot3(pose_vector[-1], pose_vector[3], pose_vector[4], pose_vector[5])
    pose_translation = Point3(pose_vector[:3])
    return Pose3(pose_rotation, pose_translation)


def create_initial_estimate(poses: List[Pose3]):
    """Create initial estimate"""
    initial_estimate = Values()
    for i, pose_vector in enumerate(poses):
        initial_estimate.insert(i, pose_vector)
    return initial_estimate


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


def generate_pose_graph(poses: List[Pose3], constraints: Iterable[Constraint]):
    """Generate pose graph."""
    graph = NonlinearFactorGraph()

    # Add the prior factor to the initial pose.
    prior_pose_factor = PriorFactorPose3(0, poses[0], noiseModel.Isotropic.Sigma(6, 5e-2))
    graph.add(prior_pose_factor)

    # Add pose priors
    for c in constraints:
        measurement_noise = noiseModel.Gaussian.Covariance(c.cov)
        factor_aTb = BetweenFactorPose3(c.a, c.b, c.aTb, measurement_noise)
        graph.add(factor_aTb)

    return graph


class TestHiltiLoader(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.loader = HiltiLoader(
            base_folder=str(TEST_DATASET_DIR_PATH),
            max_length=None,
        )

    def test_length(self) -> None:
        expected_length = 15
        self.assertEqual(len(self.loader), expected_length)

    def test_map_to_rig_idx(self) -> None:
        for i in range(0, 5, 1):
            self.assertEqual(self.loader.rig_from_image(i), 0)

        for i in range(10, 15, 1):
            self.assertEqual(self.loader.rig_from_image(i), 2)

    def test_map_to_camera_idx(self) -> None:
        for i in {0, 5, 25}:
            self.assertEqual(self.loader.camera_from_image(i), 0)

        for i in {2, 12, 32}:
            self.assertEqual(self.loader.camera_from_image(i), 2)

    def test_number_of_absolute_pose_priors(self) -> None:
        rig_idxs = list(self.loader._w_T_imu.keys())
        self.assertListEqual(rig_idxs, list(range(3)))
        num_valid_priors = 0
        for i in range(len(self.loader)):
            if self.loader.get_absolute_pose_prior(i) is not None:
                num_valid_priors += 1

        # assert no index should have pose prior
        self.assertEqual(num_valid_priors, 0)

    def test_number_of_relative_pose_priors_without_subsampling(self) -> None:
        """Check that 3 relative constraints translate into many relative pose priors."""
        expected = [
            # rig 0
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (2, 7),
            (2, 12),
            # rig 1
            (5, 7),
            (6, 7),
            (7, 8),
            (7, 9),
            (7, 12),
            # rig 2
            (10, 12),
            (11, 12),
            (12, 13),
            (12, 14),
        ]
        expected.sort()
        # Check that "stars" have been added
        relative_pose_priors = self.loader.get_relative_pose_priors()
        actual = list(relative_pose_priors.keys())
        actual.sort()
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(actual, expected)

    def test_covariance_filtered_constraints(self) -> None:
        """Check that outlier filtering works."""
        loader = HiltiLoader(
            base_folder=str(TEST_DATASET_DIR_PATH),
            max_length=None,
            subsample=2,
        )
        all_constraints = loader._load_constraints()

        expected = [(0, 1), (0, 2), (1, 2)]
        expected.sort()
        actual = [(c.a, c.b) for c in loader._covariance_filtered_constraints(all_constraints)]
        actual.sort()
        self.assertEqual(actual, expected)

    def test_number_of_relative_pose_priors_with_subsampling(self) -> None:
        """Check that 3 relative constraints translate into many relative pose priors."""
        loader = HiltiLoader(
            base_folder=str(TEST_DATASET_DIR_PATH),
            max_length=None,
            subsample=2,
        )

        expected = [
            # rig 0
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (2, 7),
            (2, 12),
            # rig 1
            (7, 12),
            # rig 2
            (10, 12),
            (11, 12),
            (12, 13),
            (12, 14),
        ]
        expected.sort()
        # Check that "stars" have been added
        relative_pose_priors = loader.get_relative_pose_priors()
        actual = list(relative_pose_priors.keys())
        actual.sort()
        self.assertEqual(actual, expected)

    def test_filters_constraints(self) -> None:
        constraints = [
            Constraint(0, 1, Pose3(Rot3(), Point3(5, 0, 0)), cov=np.zeros((6, 6))),  # outlier, has both 2 & 3 step
            Constraint(2, 1, Pose3(Rot3(), Point3(-1, 0, 0)), cov=np.zeros((6, 6))),
            Constraint(2, 3, Pose3(Rot3(), Point3(4, 0, 0)), cov=np.zeros((6, 6))),  # outlier, only 3 step
            Constraint(3, 4, Pose3(Rot3(), Point3(3, 0, 0)), cov=np.zeros((6, 6))),  # outlier, only 2 step
            Constraint(4, 5, Pose3(Rot3(), Point3(1, 0, 0)), cov=np.zeros((6, 6))),
            Constraint(2, 0, Pose3(Rot3(), Point3(-2, 0, 0)), cov=np.zeros((6, 6))),
            Constraint(1, 3, Pose3(Rot3(), Point3(2, 0, 0)), cov=np.zeros((6, 6))),
            Constraint(3, 5, Pose3(Rot3(), Point3(2, 0, 0)), cov=np.zeros((6, 6))),
            Constraint(0, 3, Pose3(Rot3(), Point3(3, 0, 0)), cov=np.zeros((6, 6))),
            Constraint(1, 4, Pose3(Rot3(), Point3(3, 0, 0)), cov=np.zeros((6, 6))),
            Constraint(2, 5, Pose3(Rot3(), Point3(3, 0, 0)), cov=np.zeros((6, 6))),
        ]

        expected_outliers = [(0, 1), (2, 3), (3, 4)]

        filtered = HiltiLoader._filter_outlier_constraints(constraints)
        outlier_keys = [(c.a, c.b) for c in filtered if c.cov[0, 0] > np.deg2rad(10)]
        self.assertSetEqual(set(outlier_keys), set(expected_outliers))

    def test_updates_stationary_constraints(self) -> None:
        constraints = [
            Constraint(0, 1, Pose3(Rot3(), Point3(5, 0, 0))),
            Constraint(2, 1, Pose3(Rot3(), Point3(0.01, 0, 0))),  # stationary
            Constraint(2, 3, Pose3(Rot3(), Point3(0, 0.01, 0))),  # stationary
            Constraint(3, 4, Pose3(Rot3(), Point3(3, 0, 0))),
            Constraint(4, 5, Pose3(Rot3.Rz(np.deg2rad(20)), Point3(0, 0, 0))),
        ]
        updated_constraints = HiltiLoader._update_stationary_constraints(constraints)
        zero_keys = [(c.a, c.b) for c in updated_constraints if c.aTb.equals(Pose3(), 1e-4)]
        self.assertSetEqual(set([(2, 1), (2, 3)]), set(zero_keys))

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
        filtered_constraints = list(HiltiLoader._covariance_filtered_constraints(constraints))
        filtered_constraints = HiltiLoader._filter_outlier_constraints(filtered_constraints)
        graph = generate_pose_graph(poses, filtered_constraints)
        self.assertEqual(graph.size(), 10328)

        # Check initial error.
        initial_estimate = create_initial_estimate(poses)
        self.assertAlmostEqual(graph.error(initial_estimate), 60807.0197569433)

        # Check that we can initialize with Pose3Initialize
        # TODO(Frank): this is still high, and an outlier is certainly present
        initial_estimate2 = InitializePose3.initialize(graph)
        self.assertAlmostEqual(graph.error(initial_estimate2), 3390522.412178782, places=1)

        # Optimize and Check final error.
        optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate2)
        result = optimizer.optimize()
        final_error = graph.error(result)
        self.assertAlmostEqual(final_error, 27083.849612680227, places=1)


if __name__ == "__main__":
    unittest.main()
