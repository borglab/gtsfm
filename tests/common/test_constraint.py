"""Unit tests for the RigRetriever.

Author: Frank Dellaert
"""

import tempfile
import unittest
from pathlib import Path

import gtsam
import numpy as np
from gtsam.utils.test_case import GtsamTestCase

from gtsfm.common.constraint import Constraint

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"


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
        path = DATA_ROOT_PATH / "test_constraints.txt"
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
        self.assertEqual(self.a_constraint_b.predicted_pairs(threshold=30), [(5, 10)])
        self.assertEqual(self.a_constraint_c.predicted_pairs(threshold=30), [(5 + 1, 15 + 2)])


if __name__ == "__main__":
    unittest.main()
