"""Unit tests for merging partitions functionality.

Authors: Richi Dubey
"""
import unittest
import numpy as np

# --- GTSAM Imports ---
# Attempt to import gtsam, provide helpful error message if missing
try:
    import gtsam
except ImportError:
    print("*" * 80)
    print("WARN: GTSAM import failed. Skipping GTSAM-dependent tests.")
    print("Please install GTSAM (e.g., 'pip install gtsam') to run these tests.")
    print("*" * 80)
    gtsam = None # Set to None so tests can be conditionally skipped

# --- Function Under Test ---
try:
    from gtsfm.runner.gtsfm_runner_base import merge_two_partition_results
    GTSFM_AVAILABLE = True
except ImportError as e:
    print(f"WARN: Could not import merge_two_partition_results: {e}")
    GTSFM_AVAILABLE = False # Set flag if import fails

# --- Helper function for comparing Pose3 dictionaries ---
def assert_pose_dicts_equal(test_case, dict1, dict2, tolerance=1e-6):
    """Asserts two dictionaries mapping int to gtsam.Pose3 are equal."""
    test_case.assertIsInstance(dict1, dict)
    test_case.assertIsInstance(dict2, dict)
    test_case.assertSetEqual(set(dict1.keys()), set(dict2.keys()))
    for key in dict1:
        test_case.assertTrue(
            dict1[key].equals(dict2[key], tolerance),
            f"Poses for key {key} differ:\n{dict1[key]}\nvs\n{dict2[key]}"
        )

# --- Test Class ---
@unittest.skipIf(gtsam is None or not GTSFM_AVAILABLE, "GTSAM not found or gtsfm_runner_base import failed")
class TestMergeTwoPartitionResults(unittest.TestCase):
    """Tests the merge_two_partition_results function."""

    def setUp(self):
        """Set up common poses for tests."""
        self.pose0 = gtsam.Pose3() # Identity
        self.pose1 = gtsam.Pose3(gtsam.Rot3.Yaw(np.pi/4), gtsam.Point3(1, 0, 0))
        self.pose2 = gtsam.Pose3(gtsam.Rot3.Roll(np.pi/6), gtsam.Point3(0, 2, 0))
        self.pose3 = gtsam.Pose3(gtsam.Rot3.Pitch(np.pi/3), gtsam.Point3(0, 0, 3))

        # Define a transformation between hypothetical frames 'a' and 'b'
        # aTb: Transformation from frame 'b' to frame 'a'
        self.aTb_translation = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(5, -5, 10))
        self.aTb_rotation = gtsam.Pose3(gtsam.Rot3.Rodrigues(0.1, 0.2, 0.3), gtsam.Point3())
        self.aTb_combined = gtsam.Pose3(gtsam.Rot3.Ypr(0.1, 0.2, 0.3), gtsam.Point3(1, 2, 3))


    def test_no_overlap(self):
        """Test merging when there are no common camera indices."""
        poses1 = {0: self.pose0, 1: self.pose1}
        poses2 = {2: self.pose2, 3: self.pose3}
        with self.assertRaisesRegex(ValueError, "No overlapping cameras found"):
            merge_two_partition_results(poses1, poses2)

    def test_perfect_overlap_identity_transform(self):
        """Test merging when partitions are identical and aTb should be identity."""
        poses1 = {0: self.pose0, 1: self.pose1, 2: self.pose2}
        poses2 = {0: self.pose0, 1: self.pose1, 2: self.pose2, 3: self.pose3} # poses2 has an extra pose

        expected_merged_poses = {
            0: self.pose0,
            1: self.pose1,
            2: self.pose2,
            3: self.pose3 # Pose 3 from poses2 should be added directly (since aTb is Identity)
        }

        merged_poses = merge_two_partition_results(poses1, poses2)
        assert_pose_dicts_equal(self, merged_poses, expected_merged_poses, tolerance=1e-7)

    def test_overlap_with_translation(self):
        """Test merging when partition 2 is translated relative to partition 1."""
        aTb = self.aTb_translation
        bTa = aTb.inverse() # Transformation from 'a' to 'b'

        # Poses in frame 'a'
        poses1 = {0: self.pose0, 1: self.pose1}
        # Poses in frame 'b' (derived from frame 'a' poses)
        poses2 = {
            0: bTa.compose(self.pose0),
            1: bTa.compose(self.pose1),
            2: bTa.compose(self.pose2) # Extra pose in partition 2
        }

        # Expected result: all poses in frame 'a'
        expected_merged_poses = {
            0: self.pose0,
            1: self.pose1,
            2: self.pose2 # aTb * (bTa * pose2) should recover pose2
        }

        merged_poses = merge_two_partition_results(poses1, poses2)
        # Optimization might have small errors
        assert_pose_dicts_equal(self, merged_poses, expected_merged_poses, tolerance=1e-6)

    def test_overlap_with_combined_transform(self):
        """Test merging with both translation and rotation offset."""
        aTb = self.aTb_combined
        bTa = aTb.inverse()

        poses1 = {1: self.pose1, 2: self.pose2} # Different indices
        poses2 = {
            1: bTa.compose(self.pose1),
            2: bTa.compose(self.pose2),
            3: bTa.compose(self.pose3) # Extra pose
        }
        expected_merged_poses = {
            1: self.pose1,
            2: self.pose2,
            3: self.pose3
        }

        merged_poses = merge_two_partition_results(poses1, poses2)
        assert_pose_dicts_equal(self, merged_poses, expected_merged_poses, tolerance=1e-6)

    def test_minimal_overlap(self):
        """Test merging with only one overlapping camera."""
        aTb = self.aTb_translation
        bTa = aTb.inverse()

        poses1 = {0: self.pose0, 99: self.pose1} # 99 won't overlap
        poses2 = {
            0: bTa.compose(self.pose0), # The only overlap
            2: bTa.compose(self.pose2)  # Extra pose
        }
        expected_merged_poses = {
            0: self.pose0,
            99: self.pose1,
            2: self.pose2 # aTb * (bTa * pose2)
        }

        merged_poses = merge_two_partition_results(poses1, poses2)
        # With only one overlap, optimization is exact for that constraint
        assert_pose_dicts_equal(self, merged_poses, expected_merged_poses, tolerance=1e-7)

    def test_empty_poses1(self):
        """Test merging when the first partition is empty."""
        poses1 = {}
        poses2 = {0: self.pose0, 1: self.pose1}
        # Should raise ValueError because there's no overlap
        with self.assertRaisesRegex(ValueError, "No overlapping cameras found"):
            merge_two_partition_results(poses1, poses2)

    def test_empty_poses2(self):
        """Test merging when the second partition is empty."""
        poses1 = {0: self.pose0, 1: self.pose1}
        poses2 = {}
        # Expected result is just poses1, as there's nothing to merge.
        # The function technically finds no overlap, but returns early
        # before raising the error if poses2 is empty. Let's check the code...
        # The loop `for k, bTi in poses2.items():` won't execute.
        # The code will successfully find `aTb` if there *was* overlap, but
        # the merging part doesn't happen. If there was *no* overlap, it errors.
        # Let's refine test cases based on this understanding.

        # Case 1: No overlap (poses1 has keys, poses2 is empty) -> Should raise error
        with self.assertRaisesRegex(ValueError, "No overlapping cameras found"):
             merge_two_partition_results({0: self.pose0}, {}) # No overlap

        # Case 2: Overlap exists (but poses2 is empty). This scenario is impossible
        # by definition (overlap requires poses2 to have common keys).
        # The code *should* raise ValueError if poses1 is non-empty and poses2 is empty.

    def test_both_empty(self):
        """Test merging when both partitions are empty."""
        poses1 = {}
        poses2 = {}
        # Should raise ValueError because there's no overlap
        with self.assertRaisesRegex(ValueError, "No overlapping cameras found"):
            merge_two_partition_results(poses1, poses2)

if __name__ == "__main__":
    unittest.main()
