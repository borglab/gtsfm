"""Unit tests for custom serialization methods.

Authors: Ayush Baid
"""
import unittest

from gtsam import Rot3

import utils.serialization as serialization_utils


class TestSerialization(unittest.TestCase):
    """Unit tests for custom serialization for GTSAM types."""

    def test_rot3_roundtrip(self):
        """Test the round-trip on Rot3 object."""

        expected = Rot3.RzRyRx(0, 0.05, 0.1)

        header, frames = serialization_utils.serialize_Rot3(expected)

        recovered = serialization_utils.deserialize_Rot3(header, frames)

        self.assertTrue(expected.equals(recovered, 1e-5))


if __name__ == '__main__':
    unittest.main()
