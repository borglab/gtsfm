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
            base_folder=str(TEST_DATASET_DIR_PATH),
            max_frame_lookahead=1,
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

    def test_is_valid_pair(self) -> None:
        # cameras in the same timestamp
        self.assertTrue(self.loader.is_valid_pair(0, 3))
        self.assertFalse(self.loader.is_valid_pair(0, 2))

        # same camera in the next timestamp
        for i in range(5):
            self.assertTrue(self.loader.is_valid_pair(i, i + 5))

        # cam0 at t=0 and cam1 at t=0
        self.assertTrue(self.loader.is_valid_pair(0, 1))

        # cam2 at t=0 and cam2 at t=2
        self.assertFalse(self.loader.is_valid_pair(2, 12))

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

    def test_get_valid_pairs(self) -> None:
        i1: int = 5  # cam 0 rig 1

        # 0, 1, 3 from other rigs
        # 1, 3 from the same rig
        expected_i2s = {
            0,  # cam 0 rig 0
            1,  # cam 1 rig 0
            3,  # cam 3 rig 0
            6,  # cam 1 rig 1
            8,  # cam 3 rig 1
            10,  # cam 0 rig 2
            11,  # cam 1 rig 2
            13,  # cam 3 rig 2
        }

        for i2 in range(len(self.loader)):
            print(i1, i2)
            if i2 in expected_i2s:
                if i1 < i2:
                    self.assertTrue(self.loader.is_valid_pair(i1, i2))
                else:
                    self.assertTrue(self.loader.is_valid_pair(i2, i1))
            else:
                self.assertFalse(self.loader.is_valid_pair(i1, i2))

    def test_number_of_relative_pose_priors(self) -> None:
        """Check that 3 relative constraints translate into many relative pose priors."""
        # The pairs we get from the rig_retriever are these:
        pairs = [
            (5, 10),
            (5, 11),
            (5, 13),
            (5, 14),
            (6, 10),
            (6, 11),
            (6, 14),
            (8, 10),
            (8, 13),
            (9, 10),
            (9, 11),
            (9, 14),
            (0, 10),
            (0, 11),
            (0, 14),
            (1, 10),
            (1, 11),
            (1, 14),
            (3, 13),
            (4, 10),
            (4, 11),
            (4, 14),
            (0, 5),
            (0, 6),
            (0, 9),
            (1, 5),
            (1, 6),
            (1, 9),
            (3, 8),
            (4, 5),
            (4, 6),
            (4, 9),
        ]
        relative_pose_priors = self.loader.get_relative_pose_priors(pairs)
        self.assertEqual(len(relative_pose_priors), len(pairs))


if __name__ == "__main__":
    unittest.main()
