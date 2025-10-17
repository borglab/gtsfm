"""Unit tests for the hilti loader.

Note: currently running on the whole dataset.
"""

import unittest
from pathlib import Path

import numpy as np

import gtsfm.utils.geometry_comparisons as comp_utils
from gtsfm.loader.hilti_loader import HiltiLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
TEST_DATASET_DIR_PATH = DATA_ROOT_PATH / "hilti_exp4_small"


class TestHiltiLoader(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.loader = HiltiLoader(
            dataset_dir=str(TEST_DATASET_DIR_PATH),
            max_length=None,
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
            else:
                print(f"not found for i={i}")

        # assert all poses have absolute pose priors
        self.assertEqual(num_valid_priors, len(self.loader))

    def test_get_absolute_pose_priors(self) -> None:
        # starting frames have no camera movement
        t0_cam0_prior = self.loader.get_absolute_pose_prior(0)
        t1_cam0_prior = self.loader.get_absolute_pose_prior(5)

        self.assertLessEqual(
            comp_utils.compute_relative_rotation_angle(t0_cam0_prior.value.rotation(), t1_cam0_prior.value.rotation()),
            2,
        )

        self.assertLessEqual(
            np.linalg.norm(t0_cam0_prior.value.translation() - t1_cam0_prior.value.translation()), 0.01
        )

        # self.assertLessEqual(
        #     comp_utils.compute_relative_unit_translation_angle(
        #         Unit3(t0_cam0_prior.value.translation()), Unit3(t1_cam0_prior.value.translation())
        #     ),
        #     2,
        # )

    def test_number_of_relative_pose_priors(self) -> None:
        """Check that 3 relative constraints translate into many relative pose priors."""
        # Just give 3 pairs
        pairs = [
            (0, 1),
            (0, 3),
            (0, 5),
        ]
        expected = [
            (0, 1),
            (0, 3),
            (0, 5),
            (2, 0),
            (2, 1),
            (2, 3),
            (2, 4),
            (7, 5),
            (7, 6),
            (7, 8),
            (7, 9),
            (12, 10),
            (12, 11),
            (12, 13),
            (12, 14),
        ]
        expected.sort()
        # Check that "stars" have been added
        relative_pose_priors = self.loader.get_relative_pose_priors(pairs)
        actual = list(relative_pose_priors.keys())
        actual.sort()
        self.assertEqual(len(actual), len(pairs) + 3 * 4)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
