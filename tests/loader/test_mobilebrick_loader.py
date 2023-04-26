"""Unit tests for the MobileBrick loader.

Authors: Akshay Krishnan
"""

import unittest
from pathlib import Path

from gtsam import Cal3Bundler, Pose3

from gtsfm.common.image import Image
from gtsfm.loader.mobilebrick_loader import MobilebrickLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"

DEFAULT_FOLDER = DATA_ROOT_PATH / "mobilebrick"


class TestMobileBrickLoader(unittest.TestCase):
    """Unit tests for the MobileBrick loader.

    The unit test data contains 5 images, their corresponding intrinsics and ground truth pose.
    """

    def setUp(self) -> None:
        """Set up the loader for the test."""
        super().setUp()

        self.loader = MobilebrickLoader(data_dir=str(DEFAULT_FOLDER))

    def test_length(self) -> None:
        """Test the number of all images in the loader."""

        # There are 5 images in total.
        self.assertEqual(5, len(self.loader))

    def test_image_filenames(self):
        """Test the image filenames."""
        image_filenames = self.loader.image_filenames()
        self.assertEqual(len(image_filenames), 5)
        self.assertEqual(str(image_filenames[0]), "000000.jpg")
        self.assertEqual(str(image_filenames[4]), "000004.jpg")

    def test_get_image_full_res(self):
        """Test the image at a given index."""
        image1 = self.loader.get_image_full_res(0)
        self.assertIsInstance(image1, Image)
        self.assertEqual(image1.shape, (1440, 1920, 3))

        with self.assertRaises(IndexError):
            self.loader.get_image_full_res(5)

    def test_get_camera_intrinsics_full_res(self):
        """Test the camera intrinsics at a given index."""
        intrinsics1 = self.loader.get_camera_intrinsics_full_res(0)
        self.assertIsInstance(intrinsics1, Cal3Bundler)

        with self.assertRaises(IndexError):
            self.loader.get_camera_intrinsics_full_res(5)

    def test_get_camera_pose(self):
        """Test the camera pose at a given index."""
        pose1 = self.loader.get_camera_pose(4)
        self.assertIsInstance(pose1, Pose3)

        with self.assertRaises(IndexError):
            self.loader.get_camera_pose(5)


if __name__ == "__main__":
    unittest.main()
