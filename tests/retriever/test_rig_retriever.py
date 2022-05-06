"""Unit tests for the RigRetriever.

Author: Frank Dellaert
"""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from cv2 import threshold

import gtsam
import numpy as np
from gtsam import Cal3Bundler, Pose3
from gtsam.utils.test_case import GtsamTestCase
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase

# from gtsfm.retriever.rig_retriever import RigRetriever

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FOLDER = DATA_ROOT_PATH / "hilti"


@dataclass
class Constraint:
    a: int = 0
    b: int = 1
    aTb: gtsam.Pose3 = gtsam.Pose3()
    cov: np.ndarray = np.eye(6)
    counts: np.ndarray = np.zeros((5, 5))

    def equals(self, other, tol) -> bool:
        """Check equality up to tolerance."""
        return (
            self.a == other.a
            and self.b == other.b
            and self.aTb.equals(other.aTb, tol)
            and np.allclose(self.cov, other.cov, atol=tol)
            and np.allclose(self.counts, other.counts)
        )

    @classmethod
    def from_row(cls, row: np.ndarray) -> "Constraint":
        """Construct from a matrix row."""
        a = int(row[0])
        b = int(row[1])
        aTb_matrix = np.vstack((row[2 : 2 + 12].reshape(3, 4), [0, 0, 0, 1]))
        aTb = gtsam.Pose3(np.round(aTb_matrix, 15))
        cov = row[14 : 14 + 36].reshape(6, 6)
        counts = row[50:].reshape(5, 5).astype(int)
        return Constraint(a, b, aTb, cov, counts)

    def to_row(self) -> np.ndarray:
        """Serialize into a single row."""
        return np.hstack(
            [
                self.a,
                self.b,
                self.aTb.matrix()[:3, :].reshape((12,)),
                self.cov.reshape((6 * 6,)),
                self.counts.reshape((25,)),
            ]
        )

    @staticmethod
    def write(fname: str, constraints: List["Constraint"]):
        """Write constraints to text file."""
        big_matrix = np.array([constraint.to_row() for constraint in constraints])
        np.savetxt(fname, big_matrix)

    @staticmethod
    def read(fname: str) -> List["Constraint"]:
        """Read constraints from text file."""
        constraint_matrix = np.loadtxt(fname)
        return [Constraint.from_row(row) for row in constraint_matrix]

    def edges(self, threshold=30):
        """Return pairs of cameras with over `threshold` count."""
        return [
            (self.a * 5 + camera_index_in_a, self.b * 5 + camera_index_in_b)
            for camera_index_in_a in range(5)
            for camera_index_in_b in range(5)
            if self.counts[camera_index_in_a, camera_index_in_b] >= threshold
        ]


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


class RigRetriever(RetrieverBase):
    """Retriever for camera rigs inspired by the Hilti challenge."""

    def __init__(self, constraints_path: Path, threshold: int = 100):
        """Initialize with path to a constraints file."""
        self._constraints_path = constraints_path
        self._threshold = threshold

    def run(self, loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        constraints = Constraint.read(str(self._constraints_path))

        return sum([c.edges(self._threshold) for c in constraints], [])


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
