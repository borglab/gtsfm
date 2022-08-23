"""Unit tests for the YFCC IMB loader.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from gtsam import Cal3Bundler, Pose3

import gtsfm.utils.io as io_utils
from gtsfm.loader.yfcc_imb_loader import YfccImbLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
FOLDER_PATH = DATA_ROOT_PATH / "imb_reichstag"


class TestYfccIMbLoader(unittest.TestCase):
    """Unit test for the YFCC IMB Loader."""

    def setUp(self) -> None:
        super().setUp()
        self.loader = YfccImbLoader(str(FOLDER_PATH))

    def test_len(self):
        """Test the number of entries in the loader."""

        expected = 10

        self.assertEqual(len(self.loader), expected)

    def test_get_image_valid_index(self):
        """Tests that get_image works for all valid indices."""

        for idx in range(len(self.loader)):
            self.assertIsNotNone(self.loader.get_image(idx))

    def test_get_image_invalid_index(self):
        """Test that get_image raises an exception on an invalid index."""

        # negative index
        with self.assertRaises(IndexError):
            self.loader.get_image(-1)
        # len() as index
        with self.assertRaises(IndexError):
            self.loader.get_image(10)
        # index > len()
        with self.assertRaises(IndexError):
            self.loader.get_image(100)

    def test_image_contents(self):
        """Test the actual image which is being fetched by the loader at an index.

        This test's primary purpose is to check if the ordering of filename is being respected by the loader
        """

        index_to_test = 5
        file_path = FOLDER_PATH / "images" / "05866831_3427466899.jpg"

        loader_image = self.loader.get_image(index_to_test)

        expected_image = io_utils.load_image(file_path)

        np.testing.assert_allclose(expected_image.value_array, loader_image.value_array)

    def test_get_camera_pose(self):
        """Tests that the correct pose is fetched."""

        index_to_test = 5

        fetched = self.loader.get_camera_pose(index_to_test)

        expected = Pose3(
            np.array(
                [
                    [
                        9.608e-01,
                        4.766e-04,
                        2.770e-01,
                        -5.205e00,
                    ],
                    [
                        -1.597e-02,
                        9.984e-01,
                        5.367e-02,
                        -1.027e00,
                    ],
                    [
                        -2.765e-01,
                        -5.600e-02,
                        9.593e-01,
                        -6.992e00,
                    ],
                    [
                        0.000e00,
                        0.000e00,
                        0.000e00,
                        1.000e00,
                    ],
                ]
            )
        )

        self.assertTrue(expected.equals(fetched, 1e-2))

    def test_get_camera_intrinsics(self):
        index_to_test = 5

        fetched = self.loader.get_camera_intrinsics(index_to_test)

        expected = Cal3Bundler(
            fx=1915.71851,
            k1=0.0,
            k2=0.0,
            u0=510.5,
            v0=362.5,
        )

        self.assertTrue(expected.equals(fetched, 1e-2))

    def test_all_internal_pairs_are_valid(self) -> None:
        pairs = self.loader._image_pairs

        for i1, i2 in pairs:
            self.assertTrue(self.loader.is_valid_pair(i1, i2))

    @patch("gtsfm.loader.loader_base.LoaderBase.is_valid_pair", return_value=True)
    def test_is_valid_pair(self, base_is_valid_pair_mock: MagicMock) -> None:
        i1 = 1
        i2 = 10
        self.loader.is_valid_pair(i1, i2)
        base_is_valid_pair_mock.assert_called_once_with(i1, i2)


if __name__ == "__main__":
    unittest.main()
