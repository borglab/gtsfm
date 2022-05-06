"""Unit tests for the RigRetriever.

Author: Frank Dellaert
"""

import tempfile
import unittest
from pathlib import Path
from typing import List, Optional

import gtsam
import numpy as np
from gtsam import Cal3Bundler, Pose3
from gtsam.utils.test_case import GtsamTestCase
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.rig_retriever import Constraint, RigRetriever

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FOLDER = DATA_ROOT_PATH / "hilti"


class HiltiLoader(LoaderBase):
    """Only reads constraints for now."""

    def __init__(self, root_path: Path):
        """Initialize with Hilti dataset directory."""
        self.constraints_file = root_path / "constraints.txt"

    def __len__(self) -> int:
        return 0

    def get_image_full_res(self, index: int) -> Image:
        return None

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Bundler]:
        return None

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        return None


class TestConstraint(GtsamTestCase):
    def setUp(self):
        """Set up two constraints to test."""
        aTb = gtsam.Pose3(gtsam.Rot3.Yaw(np.pi / 2), gtsam.Point3(1, 2, 3))
        cov = gtsam.noiseModel.Isotropic.Variance(6, 1e-3).covariance()
        counts = np.zeros((5, 5))
        counts[0][0] = 200
        self.a_constraint_b = Constraint(1, 2, aTb, cov, counts)

        aTc = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.5, 0.5, 0.5))
        variances = np.array([1e-4, 1e-5, 1e-6, 1e-1, 1e-2, 1e-3])
        aCOVc = gtsam.noiseModel.Diagonal.Variances(variances).covariance()
        counts = np.zeros((5, 5))
        counts[1][2] = 100
        self.a_constraint_c = Constraint(1, 3, aTc, aCOVc, counts)

    def test_equals(self):
        """Test we can test."""
        self.assertFalse(self.a_constraint_b.equals(self.a_constraint_c, 1e-9))

    def test_read(self):
        """Read from file in test data directory."""
        path = DEFAULT_FOLDER / "test_constraints.txt"
        actual = Constraint.read(str(path))
        expected = [self.a_constraint_b, self.a_constraint_c]
        self.gtsamAssertEquals(actual[0], expected[0], 1e-9)
        self.gtsamAssertEquals(actual[1], expected[1], 1e-9)

    def test_round_trip(self):
        """Round-trip from temporary file"""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_file = Path(tempdir) / "temp_constraints.txt"
            constraints = [self.a_constraint_b, self.a_constraint_c]
            Constraint.write(str(temp_file), constraints)
            actual = Constraint.read(temp_file)
            self.gtsamAssertEquals(actual[0], constraints[0], 1e-9)
            self.gtsamAssertEquals(actual[1], constraints[1], 1e-9)

    def test_edges(self):
        """Test creating edges between cameras."""
        self.assertEqual(self.a_constraint_b.edges(threshold=30), [(5, 10)])
        self.assertEqual(self.a_constraint_c.edges(threshold=30), [(5 + 1, 15 + 2)])


class TestRigRetriever(unittest.TestCase):
    def test_rig_retriever(self) -> None:
        """Assert that we can parse a constraints file from the Hilti SLAM team and get constraints."""

        loader = HiltiLoader(DEFAULT_FOLDER)
        constraints_path = DEFAULT_FOLDER / "constraints.txt"
        retriever = RigRetriever(constraints_path, threshold=150)

        pairs = retriever.run(loader=loader)
        self.assertEqual(len(pairs), 318)  # regression

        # regression on pairs
        expected = [(2502, 2497), (2504, 2498), (2500, 2507), (2502, 2506), (2502, 2508)]
        self.assertEqual(pairs[:5], expected)

        # That first pair above does correspond to first constraint
        constraints = Constraint.read(str(constraints_path))
        c_500_499 = constraints[0]
        self.assertEqual(c_500_499.a, 2502 // 5)
        self.assertEqual(c_500_499.b, 2497 // 5)
        self.assertTrue(c_500_499.counts[2502 % 5, 2497 % 5] >= 150)


if __name__ == "__main__":
    unittest.main()
