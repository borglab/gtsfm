"""Unit tests for custom serialization methods.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import Point3, Rot3, Unit3

import gtsfm.utils.serialization as serialization_utils


class TestSerialization(unittest.TestCase):
    """Unit tests for custom serialization for GTSAM types."""

    def test_point3_roundtrip(self):
        """Test the round-trip on Point3 object."""

        expected = Point3(np.random.randn(3))

        header, frames = serialization_utils.serialize_Point3(expected)

        recovered = serialization_utils.deserialize_Point3(header, frames)

        np.testing.assert_allclose(expected, recovered)

    def test_rot3_roundtrip(self):
        """Test the round-trip on Rot3 object."""

        expected = Rot3.RzRyRx(0, 0.05, 0.1)

        header, frames = serialization_utils.serialize_Rot3(expected)

        recovered = serialization_utils.deserialize_Rot3(header, frames)

        self.assertTrue(expected.equals(recovered, 1e-5))

    def test_unit3_roundtrip(self):
        """Test the round-trip on Unit3 object."""

        expected = Unit3(np.random.randn(3))

        header, frames = serialization_utils.serialize_Unit3(expected)

        recovered = serialization_utils.deserialize_Unit3(header, frames)

        self.assertTrue(expected.equals(recovered, 1e-5))


if __name__ == '__main__':
    unittest.main()
