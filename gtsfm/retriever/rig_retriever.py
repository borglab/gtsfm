"""Retriever for camera rigs that come with a constraints file.

Author: Frank Dellaert
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import gtsam
import numpy as np
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase


@dataclass
class Constraint:
    """A pose constraint between two poses.
    a and b are the pose indices.
    aTb and cov are the relative pose and 6x6 covariance on tangent space.
    counts[i1,i2] is predicted number of visual correspondences for cameras i1 on rig 1, i2 on rig b.
    """

    a: int
    b: int
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
        aTb_matrix = np.vstack((row[2:14].reshape(3, 4), [0, 0, 0, 1]))
        aTb = gtsam.Pose3(np.round(aTb_matrix, 15))
        cov = row[14:50].reshape(6, 6)
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