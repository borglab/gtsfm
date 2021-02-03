"""Unit tests for comparison functions for geometry types.

Authors: Ayush Baid
"""
import unittest

import numpy as np
from gtsam import Cal3_S2, Pose3, Rot3, Unit3
from gtsam.examples import SFMdata

import gtsfm.utils.geometry_comparisons as geometry_comparisons

POSE_LIST = SFMdata.createPoses(Cal3_S2())


class TestGeometryComparisons(unittest.TestCase):
    """Unit tests for comparison functions for geometry types."""

    def test_compare_poses_exact(self):
        """Check pose comparison with exactly same inputs."""
        self.assertTrue(
            geometry_comparisons.compare_global_poses(POSE_LIST, POSE_LIST)
        )

    def test_compare_poses_with_uniform_scaled_translations(self):
        """Check pose comparison with all translations in input #2 scaled by
        the same scalar factor."""
        scale_factor = 1.2
        pose_list_ = [
            Pose3(x.rotation(), x.translation() * scale_factor)
            for x in POSE_LIST
        ]

        self.assertTrue(
            geometry_comparisons.compare_global_poses(POSE_LIST, pose_list_)
        )

    def test_compare_poses_with_nonuniform_scaled_translations(self):
        """Check pose comparison with all translations in input #2 scaled by
        significantly different scalar factors."""
        scale_factors = [0.3, 0.7, 0.9, 1.0, 1.0, 0.99, 1.01, 1.10]
        pose_list_ = [
            Pose3(x.rotation(), x.translation() * scale_factors[idx])
            for idx, x in enumerate(POSE_LIST)
        ]

        self.assertFalse(
            geometry_comparisons.compare_global_poses(POSE_LIST, pose_list_)
        )

    def test_compare_poses_with_origin_shift(self):
        """Check pose comparison with a shift in the global origin."""
        new_origin = Pose3(
            Rot3.RzRyRx(0.3, 0.1, -0.27), np.array([-20.0, +19.0, 3.5])
        )

        pose_list_ = [new_origin.between(x) for x in POSE_LIST]

        self.assertTrue(
            geometry_comparisons.compare_global_poses(POSE_LIST, pose_list_)
        )

    def test_compare_different_poses(self):
        """Compare pose comparison with different inputs."""

        pose_list = [POSE_LIST[1], POSE_LIST[2], POSE_LIST[3]]
        pose_list_ = [POSE_LIST[2], POSE_LIST[3], POSE_LIST[1]]

        self.assertFalse(
            geometry_comparisons.compare_global_poses(pose_list, pose_list_)
        )

    def test_compute_relative_rotation_angle(self):
        """Tests the relative angle between two rotations."""

        R_1 = Rot3.RzRyRx(0, np.deg2rad(45), np.deg2rad(22.5))
        R_2 = Rot3.RzRyRx(0, np.deg2rad(90), np.deg2rad(22.5))

        computed = geometry_comparisons.compute_relative_rotation_angle(
            R_1, R_2
        )

        expected = np.deg2rad(45)

        np.testing.assert_allclose(computed, expected, rtol=1e-3, atol=1e-3)

    def test_compute_relative_unit_translation_angle(self):
        """Tests the relative angle between two unit-translations."""

        U_1 = Unit3(np.array([1, 0, 0]))
        U_2 = Unit3(np.array([0.5, 0.5, 0]))

        computed = geometry_comparisons.compute_relative_unit_translation_angle(
            U_1, U_2
        )

        expected = np.deg2rad(45)

        self.assertAlmostEqual(computed, expected, places=3)


if __name__ == "__main__":
    unittest.main()
