"""Unit test for the Folder Loader class.

Authors:Ayush Baid
"""
import unittest
from pathlib import Path

import dask
import numpy as np
from gtsam import Cal3Bundler, Pose3

import gtsfm.utils.io as io_utils
from gtsfm.loader.folder_loader import FolderLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"

DEFAULT_FOLDER = DATA_ROOT_PATH / "set1_1_lund_door"
EXIF_FOLDER = DATA_ROOT_PATH / "set2_lund_door_nointrinsics"
NO_EXTRINSICS_FOLDER = DATA_ROOT_PATH / "set3_lund_doornointrinsics_noextrinsics"
NO_EXIF_FOLDER = DATA_ROOT_PATH / "set4_lund_door_nointrinsics_noextrinsics_noexif"


class TestFolderLoader(unittest.TestCase):
    """Unit tests for folder loader, which loads image from a folder on disk."""

    def setUp(self):
        """Set up the loader for the test."""
        super().setUp()

        self.loader = FolderLoader(str(DEFAULT_FOLDER), image_extension="JPG")

    def test_len(self):
        """Test the number of entries in the loader."""

        self.assertEqual(12, len(self.loader))

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
            self.loader.get_image(12)
        # index > len()
        with self.assertRaises(IndexError):
            self.loader.get_image(15)

    def test_image_contents(self):
        """Test the actual image which is being fetched by the loader at an index.

        This test's primary purpose is to check if the ordering of filename is being respected by the loader
        """

        index_to_test = 5
        file_path = DEFAULT_FOLDER / "images" / "DSC_0006.JPG"

        loader_image = self.loader.get_image(index_to_test)

        expected_image = io_utils.load_image(file_path)

        np.testing.assert_allclose(expected_image.value_array, loader_image.value_array)

    def test_get_camera_pose_exists(self):
        """Tests that the correct pose is fetched (present on disk)."""

        fetched_pose = self.loader.get_camera_pose(5)

        expected_pose = Pose3(
            np.array(
                [
                    # [0.9387, 0.0592, 0.3510, -4.5075],
                    # [-0.0634, 1.0043, -0.01437, 0.2307],
                    # [-0.3618, -0.0227, 0.9362, 1.4820],
                    # [0.0, 0.0, 0.0, 1.0],
                    [ 0.93414109,  0.06411186,  0.35109842, -4.4532169 ],
                    [-0.06308349,  0.99790466, -0.01437957,  0.22408744],
                    [-0.35128465, -0.00871597,  0.93622814,  1.41870188],
                    [ 0.        ,  0.        ,  0.        ,  1.        ],
                ]
            )
        )

        self.assertTrue(expected_pose.equals(fetched_pose, 1e-2))

    def test_get_camera_pose_missing(self):
        """Tests that the camera pose is None, because it is missing on disk."""

        loader = FolderLoader(str(NO_EXTRINSICS_FOLDER), image_extension="JPG")

        fetched_pose = loader.get_camera_pose(5)

        self.assertIsNone(fetched_pose)

    def test_get_camera_intrinsics_explicit(self):
        """Tests getter for intrinsics when explicit numpy arrays with intrinsics are present on disk."""

        computed = self.loader.get_camera_intrinsics(5)

        # expected = Cal3Bundler(fx=2378.983, k1=0, k2=0, u0=968.0, v0=648.0)
        expected = Cal3Bundler(fx=2398.118, k1=0, k2=0, u0=628.26, v0=932.38)

        self.assertTrue(expected.equals(computed, 1e-3))

    def test_get_camera_intrinsics_exif(self):
        """Tests getter for intrinsics when explicit numpy arrays are absent and we fall back on exif."""

        loader = FolderLoader(EXIF_FOLDER, image_extension="JPG")

        computed = loader.get_camera_intrinsics(5)

        expected = Cal3Bundler(fx=2378.983, k1=0, k2=0, u0=648.0, v0=968.0)
        # expected = Cal3Bundler(fx=2398.118, k1=0, k2=0, u0=628.26, v0=932.38)


        self.assertTrue(expected.equals(computed, 1e-3))

    def test_get_camera_intrinsics_missing(self):
        """Tests getter for intrinsics when explicit numpy arrays are absent and we fall back on exif."""

        loader = FolderLoader(NO_EXIF_FOLDER, image_extension="JPG")

        computed = loader.get_camera_intrinsics(5)

        self.assertIsNone(computed)

    def test_create_computation_graph_for_images(self):
        """Tests the graph for loading all the images."""

        image_graph = self.loader.create_computation_graph_for_images()

        # check the length of the graph
        self.assertEqual(12, len(image_graph))

        results = dask.compute(image_graph)[0]

        # randomly check image loads from a few indices
        np.testing.assert_allclose(results[5].value_array, self.loader.get_image(5).value_array)

        np.testing.assert_allclose(results[7].value_array, self.loader.get_image(7).value_array)

    def test_create_computation_graph_for_intrinsics(self):
        """Tests the graph for all intrinsics."""

        intrinsics_graph = self.loader.create_computation_graph_for_intrinsics()

        # check the length of the graph
        self.assertEqual(12, len(intrinsics_graph))

        results = dask.compute(intrinsics_graph)[0]

        # randomly check intrinsics from a few indices
        self.assertTrue(self.loader.get_camera_intrinsics(5).equals(results[5], 1e-5))
        self.assertTrue(self.loader.get_camera_intrinsics(7).equals(results[7], 1e-5))


if __name__ == "__main__":
    unittest.main()
