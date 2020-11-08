"""Unit test for the Folder Loader class.

Authors:Ayush Baid
"""
import unittest
from pathlib import Path

import dask
import numpy as np
from gtsam import Cal3Bundler, Pose3

import utils.io as io_utils
from loader.folder_loader import FolderLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / 'data'

DEFAULT_FOLDER = DATA_ROOT_PATH / 'set1'
EXIF_FOLDER = DATA_ROOT_PATH / 'set2_nointrinsics'
NO_EXTRINSICS_FOLDER = DATA_ROOT_PATH / 'set3_nointrinsics_noextrinsics'
NO_EXIF_FOLDER = DATA_ROOT_PATH / 'set4_nointrinsics_noextrinsics_noexif'


class TestFolderLoader(unittest.TestCase):
    """
    Unit tests for folder loader, which loads image from a folder on disk.
    """

    def setUp(self):
        """Set up the loader for the test."""
        super().setUp()

        self.loader = FolderLoader(
            str(DEFAULT_FOLDER), image_extension='JPG')

    def test_len(self):
        """
        Test the number of entries in the loader.
        """

        self.assertEqual(12, len(self.loader))

    def test_get_image_valid_index(self):
        """
        Tests that get_image works for all valid indices.
        """

        for idx in range(len(self.loader)):
            self.assertIsNotNone(self.loader.get_image(idx))

    def test_get_image_invalid_index(self):
        """
        Test that get_image raises an exception on an invalid index.
        """

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
        """
        Test the actual image which is being fetched by the loader at an index.

        This test's primary purpose is to check if the ordering of filename is being respected by the loader
        """

        index_to_test = 5
        file_path = 'tests/data/set1/images/DSC_0006.JPG'

        loader_image = self.loader.get_image(index_to_test)

        expected_image = io_utils.load_image(file_path)

        np.testing.assert_allclose(
            expected_image.value_array, loader_image.value_array)

    def test_get_camera_pose_exists(self):
        """
        Tests that the correct pose is fetched (present on disk).
        """

        fetched_pose = self.loader.get_camera_pose(5)

        expected_pose = Pose3(np.array([
            [2.46076391e+03, -1.60315873e+02, -2.54222855e+02, 1.13549066e+04],
            [4.80838640e+02,  2.37552876e+03,  8.52056819e+02, 4.00138000e+02],
            [3.51098417e-01, -1.43795686e-02,  9.36228140e-01, 2.38511070e-01]]
        ))

        self.assertTrue(expected_pose.equals(fetched_pose, 1e-5))

    def test_get_camera_pose_missing(self):
        """
        Tests that the camera pose is None, because it is missing on disk.
        """

        loader = FolderLoader(str(NO_EXTRINSICS_FOLDER), image_extension='JPG')

        fetched_pose = loader.get_camera_pose(5)

        self.assertIsNone(fetched_pose)

    def test_get_camera_intrinsics_explicit(self):
        """Tests getter for intrinsics when explicit numpy arrays with
        intrinsics are present on disk."""

        computed = self.loader.get_camera_intrinsics(5)

        expected = Cal3Bundler(fx=2378.983, k1=0, k2=0, u0=968.0, v0=648.0)

        self.assertTrue(expected.equals(computed, 1e-3))

    def test_get_camera_intrinsics_exif(self):
        """Tests getter for intrinsics when explicit numpy arrays are absent and we fall back on exif."""

        loader = FolderLoader(EXIF_FOLDER, image_extension='JPG')

        computed = loader.get_camera_intrinsics(5)

        expected = Cal3Bundler(fx=2378.983, k1=0, k2=0, u0=968.0, v0=648.0)

        self.assertTrue(expected.equals(computed, 1e-3))

    def test_get_camera_intrinsics_missing(self):
        """Tests getter for intrinsics when explicit numpy arrays are absent and we fall back on exif."""

        loader = FolderLoader(NO_EXIF_FOLDER, image_extension='JPG')

        computed = loader.get_camera_intrinsics(5)

        self.assertIsNone(computed)

    def test_delayed_get_image(self):
        """
        Checks that the delayed get API functions exactly as the normal get API
        """

        index_to_test = 5

        delayed_result = self.loader.delayed_get_image(index_to_test).compute()

        normal_result = self.loader.get_image(index_to_test)

        np.testing.assert_allclose(
            normal_result.value_array, delayed_result.value_array)

    def test_create_computation_graph(self):
        """
        Tests the graph for loading all the images
        """

        loader_graph = self.loader.create_computation_graph()

        # check the length of the graph
        self.assertEqual(12, len(loader_graph))

        results = dask.compute(loader_graph)[0]

        # randomly check image loads from a few indices
        np.testing.assert_allclose(
            results[5].value_array, self.loader.get_image(5).value_array
        )

        np.testing.assert_allclose(
            results[7].value_array, self.loader.get_image(7).value_array)
