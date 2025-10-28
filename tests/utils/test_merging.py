"""Unit tests for merging clusters functionality.

Authors: Richi Dubey
"""

import unittest

import numpy as np
from gtsam import Point3, Pose3, Rot3  # type: ignore

from gtsfm.utils.alignment import estimate_se3_from_pose_maps
from gtsfm.utils.merging import merge_pose_maps


def assert_pose_dicts_equal(test_case, dict1, dict2, tolerance=1e-6):
    """Asserts two dictionaries mapping int to Pose3 are equal."""
    test_case.assertIsInstance(dict1, dict)
    test_case.assertIsInstance(dict2, dict)
    test_case.assertSetEqual(set(dict1.keys()), set(dict2.keys()))
    for key in dict1:
        test_case.assertTrue(
            dict1[key].equals(dict2[key], tolerance), f"Poses for key {key} differ:\n{dict1[key]}\nvs\n{dict2[key]}"
        )


class TestMergeTwoClusters(unittest.TestCase):
    """Tests the merge_pose_maps function."""

    def setUp(self):
        """Set up common poses for tests."""
        self.pose0 = Pose3()  # Identity
        self.pose1 = Pose3(Rot3.Yaw(np.pi / 4), Point3(1, 0, 0))
        self.pose2 = Pose3(Rot3.Roll(np.pi / 6), Point3(0, 2, 0))
        self.pose3 = Pose3(Rot3.Pitch(np.pi / 3), Point3(0, 0, 3))

        # Define a transformation between hypothetical frames 'a' and 'b'
        # aTb: Transformation to frame 'a' from frame 'b'
        self.aTb_translation = Pose3(Rot3(), Point3(5, -5, 10))
        self.aTb_rotation = Pose3(Rot3.Rodrigues(0.1, 0.2, 0.3), Point3())
        self.aTb_combined = Pose3(Rot3.Ypr(0.1, 0.2, 0.3), Point3(1, 2, 3))

    def test_no_overlap(self):
        """Test merging when there are no common camera indices."""
        poses1 = {0: self.pose0, 1: self.pose1}
        poses2 = {2: self.pose2, 3: self.pose3}
        merged_poses = merge_pose_maps(poses1, poses2, Pose3())
        expected = {**poses1, 2: self.pose2, 3: self.pose3}
        assert_pose_dicts_equal(self, merged_poses, expected, tolerance=1e-7)

    def test_perfect_overlap_identity_transform(self):
        """Test merging when clusters are identical and aTb should be identity."""
        poses1 = {0: self.pose0, 1: self.pose1, 2: self.pose2}
        poses2 = {0: self.pose0, 1: self.pose1, 2: self.pose2, 3: self.pose3}  # poses2 has an extra pose

        expected_merged_poses = {
            0: self.pose0,
            1: self.pose1,
            2: self.pose2,
            3: self.pose3,  # Pose 3 from poses2 should be added directly (since aTb is Identity)
        }

        aTb = estimate_se3_from_pose_maps(poses1, poses2)
        merged_poses = merge_pose_maps(poses1, poses2, aTb)
        assert_pose_dicts_equal(self, merged_poses, expected_merged_poses, tolerance=1e-7)

    def test_overlap_with_translation(self):
        """Test merging when partition 2 is translated relative to partition 1."""
        aTb = self.aTb_translation
        bTa = aTb.inverse()  # Transformation from 'a' to 'b'

        # Poses in frame 'a'
        poses1 = {0: self.pose0, 1: self.pose1}
        # Poses in frame 'b' (derived from frame 'a' poses)
        poses2 = {
            0: bTa.compose(self.pose0),
            1: bTa.compose(self.pose1),
            2: bTa.compose(self.pose2),  # Extra pose in partition 2
        }

        # Expected result: all poses in frame 'a'
        expected_merged_poses = {
            0: self.pose0,
            1: self.pose1,
            2: self.pose2,  # aTb * (bTa * pose2) should recover pose2
        }

        merged_poses = merge_pose_maps(poses1, poses2, aTb)
        # Optimization might have small errors
        assert_pose_dicts_equal(self, merged_poses, expected_merged_poses, tolerance=1e-6)

    def test_overlap_with_combined_transform(self):
        """Test merging with both translation and rotation offset."""
        aTb = self.aTb_combined
        bTa = aTb.inverse()

        poses1 = {1: self.pose1, 2: self.pose2}  # Different indices
        poses2 = {1: bTa.compose(self.pose1), 2: bTa.compose(self.pose2), 3: bTa.compose(self.pose3)}  # Extra pose
        expected_merged_poses = {1: self.pose1, 2: self.pose2, 3: self.pose3}

        merged_poses = merge_pose_maps(poses1, poses2, aTb)
        assert_pose_dicts_equal(self, merged_poses, expected_merged_poses, tolerance=1e-6)

    def test_minimal_overlap(self):
        """Test merging with only one overlapping camera."""
        aTb = self.aTb_translation
        bTa = aTb.inverse()

        poses1 = {0: self.pose0, 99: self.pose1}  # 99 won't overlap
        poses2 = {0: bTa.compose(self.pose0), 2: bTa.compose(self.pose2)}  # The only overlap  # Extra pose
        expected_merged_poses = {0: self.pose0, 99: self.pose1, 2: self.pose2}  # aTb * (bTa * pose2)

        merged_poses = merge_pose_maps(poses1, poses2, aTb)
        # With only one overlap, optimization is exact for that constraint
        assert_pose_dicts_equal(self, merged_poses, expected_merged_poses, tolerance=1e-7)

    def test_empty_poses1(self):
        """Test merging when the first partition is empty."""
        poses1 = {}
        poses2 = {0: self.pose0, 1: self.pose1}
        merged_poses = merge_pose_maps(poses1, poses2, Pose3())
        assert_pose_dicts_equal(self, merged_poses, poses2, tolerance=1e-7)

    def test_empty_poses2(self):
        """Test merging when the second partition is empty."""
        poses1 = {0: self.pose0}
        merged_poses = merge_pose_maps(poses1, {}, Pose3())
        assert_pose_dicts_equal(self, merged_poses, poses1, tolerance=1e-7)

    def test_both_empty(self):
        """Test merging when both clusters are empty."""
        poses1 = {}
        poses2 = {}
        merged_poses = merge_pose_maps(poses1, poses2, Pose3())
        self.assertEqual(merged_poses, {})


if __name__ == "__main__":
    unittest.main()
