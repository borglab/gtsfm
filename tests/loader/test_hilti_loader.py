"""Unit tests for the hilti loader.

Note: currently running on the whole dataset.
"""
import unittest
from pathlib import Path

from gtsfm.loader.hilti_loader import HiltiLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
TEST_DATASET_DIR_PATH = DATA_ROOT_PATH / "hilti_exp4_small"


class TestHiltiLoader(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.loader = HiltiLoader(
            base_folder=str(TEST_DATASET_DIR_PATH),
            max_length=None,
            old_style=True,
        )

    def test_length(self) -> None:
        expected_length = 15
        self.assertEqual(len(self.loader), expected_length)

    def test_map_to_rig_idx(self) -> None:
        for i in range(0, 5, 1):
            self.assertEqual(self.loader.rig_from_image(i), 0)

        for i in range(10, 15, 1):
            self.assertEqual(self.loader.rig_from_image(i), 2)

    def test_map_to_camera_idx(self) -> None:
        for i in {0, 5, 25}:
            self.assertEqual(self.loader.camera_from_image(i), 0)

        for i in {2, 12, 32}:
            self.assertEqual(self.loader.camera_from_image(i), 2)

    def test_number_of_absolute_pose_priors(self) -> None:
        rig_idxs = list(self.loader._w_T_imu.keys())
        self.assertListEqual(rig_idxs, list(range(3)))
        num_valid_priors = 0
        for i in range(len(self.loader)):
            if self.loader.get_absolute_pose_prior(i) is not None:
                num_valid_priors += 1

        # assert no index should have pose prior
        self.assertEqual(num_valid_priors, 0)

    def test_number_of_relative_pose_priors_without_subsampling(self) -> None:
        """Check that 3 relative constraints translate into many relative pose priors."""
        expected = [
            # rig 0
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (2, 7),
            (2, 12),
            # rig 1
            (5, 7),
            (6, 7),
            (7, 8),
            (7, 9),
            (7, 12),
            # rig 2
            (10, 12),
            (11, 12),
            (12, 13),
            (12, 14),
        ]
        expected.sort()
        # Check that "stars" have been added
        relative_pose_priors = self.loader.get_relative_pose_priors()
        actual = list(relative_pose_priors.keys())
        actual.sort()
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(actual, expected)

    def test_number_of_relative_pose_priors_with_subsampling(self) -> None:
        """Check that 3 relative constraints translate into many relative pose priors."""
        loader = HiltiLoader(
            base_folder=str(TEST_DATASET_DIR_PATH),
            max_length=None,
            subsample=2,
            old_style=True,
        )

        expected = [
            # rig 0
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (2, 7),
            (2, 12),
            # rig 1
            (7, 12),
            # rig 2
            (10, 12),
            (11, 12),
            (12, 13),
            (12, 14),
        ]
        expected.sort()
        # Check that "stars" have been added
        relative_pose_priors = loader.get_relative_pose_priors()
        actual = list(relative_pose_priors.keys())
        actual.sort()
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
