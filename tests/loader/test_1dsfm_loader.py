"""Unit tests for the OneDSFM Loader class.

Authors: Yanwei Du
"""

import unittest
from pathlib import Path

from gtsfm.loader.one_d_sfm_loader import OneDSFMLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"

DEFAULT_FOLDER = DATA_ROOT_PATH / "1dsfm"


class TestOneDSfMFolderLoader(unittest.TestCase):
    """Unit tests for folder loader, which loads image from a folder on disk.

    The unit test data contains four images, one(219732_f37b838202_o.jpg) of them does not have valid exif and will
    not be loaded.
    """

    def setUp(self) -> None:
        """Set up the loader for the test."""
        super().setUp()

        self.loader = OneDSFMLoader(str(DEFAULT_FOLDER))

    def test_get_num_all_imgs(self) -> None:
        """Test the number of all images in the loader."""

        # There are four images in total.
        self.assertEqual(4, self.loader.get_num_all_imgs())

    def test_get_num_exif_imgs(self) -> None:
        """Test the number of all valid images in the loader."""

        # Only three of the four images have valid exif. Image 219732_f37b838202_o.jpg without exif is droppped.
        self.assertEqual(3, self.loader.get_num_exif_imgs())

    def test_len(self) -> None:
        """Test the number of entries in the loader."""

        # Only three of the four images have valid exif. Image 219732_f37b838202_o.jpg without exif is droppped.
        self.assertEqual(3, len(self.loader))

    def test_get_image_valid_index(self) -> None:
        """Tests that get_image works for all valid indices."""

        for idx in range(len(self.loader)):
            self.assertIsNotNone(self.loader.get_image(idx))

    def test_get_image_invalid_index(self) -> None:
        """Test that get_image raises an exception on an invalid index."""

        # negative index
        with self.assertRaises(IndexError):
            self.loader.get_image(-1)
        # len() as index
        with self.assertRaises(IndexError):
            self.loader.get_image(12)
        # index > len()
        with self.assertRaises(IndexError):
            self.loader.get_image(15)

    def test_get_camera_pose_none(self):
        """Test that the camera pose is None."""

        for idx in range(len(self.loader)):
            self.assertIsNone(self.loader.get_camera_pose(idx))

    def test_set_max_num_imgs(self):
        """Test that max_num_imgs works when loading images."""

        # 0 < max_num_imgs <= num_exif_imgs(=3)
        loader = OneDSFMLoader(str(DEFAULT_FOLDER), max_num_imgs=2)
        self.assertEqual(2, len(loader))

        # max_num_imgs > num_exif_imgs(=3)
        loader = OneDSFMLoader(str(DEFAULT_FOLDER), max_num_imgs=4)
        self.assertEqual(3, len(loader))

        # max_num_imgs = 0
        loader = OneDSFMLoader(str(DEFAULT_FOLDER), max_num_imgs=0)
        self.assertEqual(3, len(loader))

        # max_num_imgs < 0
        loader = OneDSFMLoader(str(DEFAULT_FOLDER), max_num_imgs=-1)
        self.assertEqual(3, len(loader))


if __name__ == "__main__":
    unittest.main()
