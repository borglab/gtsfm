"""Unit tests for custom serialization methods.

Authors: Ayush Baid
"""
import unittest

import gtsam
import numpy as np
from gtsam import (
    Cal3Bundler,
    PinholeCameraCal3Bundler,
    Point3,
    Pose3,
    Rot3,
    Unit3,
)

import gtsfm.utils.serialization as serialization_utils
from gtsfm.common.sfm_result import SfmResult

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_DATA = gtsam.readBal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestSerialization(unittest.TestCase):
    """Unit tests for custom serialization for GTSAM types."""

    def test_point3_roundtrip(self):
        """Test the round-trip on Point3 object."""

        expected = Point3(np.random.randn(3))

        header, frames = serialization_utils.serialize_Point3(expected)

        recovered = serialization_utils.deserialize_Point3(header, frames)

        np.testing.assert_allclose(expected, recovered)

    def test_pose3_roundtrip(self):
        """Test the round-trip on Point3 object."""

        expected = Pose3(Rot3.RzRyRx(0, 0.1, 0.2), np.random.randn(3))

        header, frames = serialization_utils.serialize_Pose3(expected)

        recovered = serialization_utils.deserialize_Pose3(header, frames)

        self.assertTrue(recovered.equals(expected, 1e-5))

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

    def test_cal3Bundler_roundtrip(self):
        """Test the round-trip on Cal3Bundler object."""

        expected = Cal3Bundler(fx=100, k1=0.1, k2=0.2, u0=100, v0=70)

        header, frames = serialization_utils.serialize_Cal3Bundler(expected)

        recovered = serialization_utils.deserialize_Cal3Bundler(header, frames)

        self.assertTrue(expected.equals(recovered, 1e-5))

    def test_pinholeCameraCal3Bundler_roundtrip(self):
        """Test the round-trip on Unit3 object."""

        expected = PinholeCameraCal3Bundler(
            Pose3(Rot3.RzRyRx(0, 0.1, -0.05), np.random.randn(3, 1)),
            Cal3Bundler(fx=100, k1=0.1, k2=0.2, u0=100, v0=70),
        )

        header, frames = serialization_utils.serialize_PinholeCameraCal3Bundler(
            expected
        )

        recovered = serialization_utils.deserialize_PinholeCameraCal3Bundler(
            header, frames
        )

        self.assertTrue(expected.equals(recovered, 1e-5))

    def test_sfmData_roundtrip(self):

        expected = EXAMPLE_DATA

        header, frames = serialization_utils.serialize_SfmData(expected)

        recovered = serialization_utils.deserialize_SfmData(header, frames)

        # comparing cameras and total reprojection error
        # self.assertListEqual(recovered.cameras, expected.cameras)

        # comparing tracks in an order-sensitive fashion.
        self.assertTrue(recovered.equals(expected, 1e-9))

    def test_sfmResult_roundtrip(self):

        expected = SfmResult(EXAMPLE_DATA, total_reproj_error=1.5)

        header, frames = serialization_utils.serialize_SfmResult(expected)

        recovered = serialization_utils.deserialize_SfmResult(header, frames)

        # comparing cameras and total reprojection error
        self.assertEqual(
            recovered.total_reproj_error, expected.total_reproj_error
        )

        self.assertTrue(recovered.sfm_data.equals(expected.sfm_data, 1e-9))


if __name__ == "__main__":
    unittest.main()
