"""Unit tests for custom serialization methods.

Authors: Ayush Baid
"""

import unittest

import gtsam
import numpy as np
from distributed.protocol.serialize import deserialize, serialize
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, Unit3

from gtsfm.common.gtsfm_data import GtsfmData

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_DATA = GtsfmData.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestSerialization(unittest.TestCase):
    """Unit tests for custom serialization for GTSAM types."""

    def test_point3_roundtrip(self):
        """Test the round-trip on Point3 object."""
        expected = Point3(np.random.randn(3))
        header, frames = serialize(expected)
        recovered = deserialize(header, frames)
        np.testing.assert_allclose(expected, recovered)

    def test_pose3_roundtrip(self):
        """Test the round-trip on Point3 object."""
        expected = Pose3(Rot3.RzRyRx(0, 0.1, 0.2), np.random.randn(3))
        header, frames = serialize(expected)
        recovered = deserialize(header, frames)
        self.assertTrue(recovered.equals(expected, 1e-5))

    def test_rot3_roundtrip(self):
        """Test the round-trip on Rot3 object."""
        expected = Rot3.RzRyRx(0, 0.05, 0.1)
        header, frames = serialize(expected)
        recovered = deserialize(header, frames)
        self.assertTrue(expected.equals(recovered, 1e-5))

    def test_unit3_roundtrip(self):
        """Test the round-trip on Unit3 object."""
        expected = Unit3(np.random.randn(3))
        header, frames = serialize(expected)
        recovered = deserialize(header, frames)
        self.assertTrue(expected.equals(recovered, 1e-5))

    def test_cal3Bundler_roundtrip(self):
        """Test the round-trip on Cal3Bundler object."""
        expected = Cal3Bundler(fx=100, k1=0.1, k2=0.2, u0=100, v0=70)
        header, frames = serialize(expected)
        recovered = deserialize(header, frames)
        self.assertTrue(expected.equals(recovered, 1e-5))

    def test_pinholeCameraCal3Bundler_roundtrip(self):
        """Test the round-trip on Unit3 object."""

        expected = PinholeCameraCal3Bundler(
            Pose3(Rot3.RzRyRx(0, 0.1, -0.05), np.random.randn(3, 1)),
            Cal3Bundler(fx=100, k1=0.1, k2=0.2, u0=100, v0=70),
        )
        header, frames = serialize(expected)
        recovered = deserialize(header, frames)

        self.assertTrue(expected.equals(recovered, 1e-5))

    def test_gtsfmData_roundtrip(self):
        """Test for equality after serializing and then de-serializing an SfmData instance."""
        expected = EXAMPLE_DATA
        header, frames = serialize(expected)
        recovered = deserialize(header, frames)

        # comparing tracks in an order-sensitive fashion.
        self.assertEqual(recovered, expected)


if __name__ == "__main__":
    unittest.main()
