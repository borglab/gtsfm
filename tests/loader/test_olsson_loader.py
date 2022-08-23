"""Unit tests for the Olsson Loader class.

Authors: John Lambert
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import dask
import numpy as np
import pytest
from gtsam import Cal3Bundler, Pose3, Rot3

import gtsfm.utils.io as io_utils
from gtsfm.loader.olsson_loader import OlssonLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"

DEFAULT_FOLDER = DATA_ROOT_PATH / "set1_lund_door"
EXIF_FOLDER = DATA_ROOT_PATH / "set2_lund_door_nointrinsics"
NO_EXTRINSICS_FOLDER = DATA_ROOT_PATH / "set3_lund_doornointrinsics_noextrinsics"
NO_EXIF_FOLDER = DATA_ROOT_PATH / "set4_lund_door_nointrinsics_noextrinsics_noexif"


class TestFolderLoader(unittest.TestCase):
    """Unit tests for folder loader, which loads image from a folder on disk."""

    def setUp(self) -> None:
        """Set up the loader for the test."""
        super().setUp()

        self.loader = OlssonLoader(str(DEFAULT_FOLDER), image_extension="JPG", max_frame_lookahead=4)

    def test_len(self) -> None:
        """Test the number of entries in the loader."""

        self.assertEqual(12, len(self.loader))

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

    def test_image_contents(self) -> None:
        """Test the actual image which is being fetched by the loader at an index.

        This test's primary purpose is to check if the ordering of filename is being respected by the loader
        """
        index_to_test = 5
        file_path = DEFAULT_FOLDER / "images" / "DSC_0006.JPG"
        loader_image = self.loader.get_image_full_res(index_to_test)
        expected_image = io_utils.load_image(file_path)
        np.testing.assert_allclose(expected_image.value_array, loader_image.value_array)

    def test_get_camera_pose_exists(self) -> None:
        """Tests that the correct pose is fetched (present on disk)."""
        fetched_pose = self.loader.get_camera_pose(1)

        wRi_expected = np.array(
            [
                [0.998079, 0.015881, 0.0598844],
                [-0.0161175, 0.999864, 0.00346851],
                [-0.0598212, -0.00442703, 0.998199],
            ]
        )
        wti_expected = np.array([-0.826311, -0.00409053, 0.111315])

        expected_pose = Pose3(Rot3(wRi_expected), wti_expected)
        self.assertTrue(expected_pose.equals(fetched_pose, 1e-2))

    def test_get_camera_pose_missing(self):
        """Tests that the camera pose is None, because it is missing on disk."""
        loader = OlssonLoader(str(NO_EXTRINSICS_FOLDER), image_extension="JPG")
        fetched_pose = loader.get_camera_pose(5)
        self.assertIsNone(fetched_pose)

    def test_get_camera_intrinsics_explicit(self) -> None:
        """Tests getter for intrinsics when explicit data.mat file with intrinsics are present on disk."""
        expected_fx = 2398.119
        expected_fy = 2393.952
        expected_fx = min(expected_fx, expected_fy)

        expected_px = 628.265
        expected_py = 932.382

        computed = self.loader.get_camera_intrinsics_full_res(5)
        expected = Cal3Bundler(fx=expected_fx, k1=0, k2=0, u0=expected_px, v0=expected_py)

        self.assertTrue(expected.equals(computed, 1e-3))

    def test_get_camera_intrinsics_exif(self) -> None:
        """Tests getter for intrinsics when explicit numpy arrays are absent and we fall back on exif."""
        loader = OlssonLoader(EXIF_FOLDER, image_extension="JPG", use_gt_intrinsics=False)
        computed = loader.get_camera_intrinsics_full_res(5)
        expected = Cal3Bundler(fx=2378.983, k1=0, k2=0, u0=648.0, v0=968.0)
        self.assertTrue(expected.equals(computed, 1e-3))

    def test_get_camera_intrinsics_missing(self) -> None:
        """Tests getter for intrinsics when explicit numpy arrays are absent, exif is missing, and we raise an error."""
        loader = OlssonLoader(NO_EXIF_FOLDER, image_extension="JPG")
        with pytest.raises(ValueError):
            _ = loader.get_camera_intrinsics(5)

    def test_create_computation_graph_for_images(self) -> None:
        """Tests the graph for loading all the images."""
        image_graph = self.loader.create_computation_graph_for_images()

        # check the length of the graph
        self.assertEqual(12, len(image_graph))
        results = dask.compute(image_graph)[0]

        # randomly check image loads from a few indices
        np.testing.assert_allclose(results[5].value_array, self.loader.get_image(5).value_array)
        np.testing.assert_allclose(results[7].value_array, self.loader.get_image(7).value_array)

    def test_get_all_intrinsics(self) -> None:
        """Tests the graph for all intrinsics."""

        all_intrinsics = self.loader.get_all_intrinsics()

        # check the length of the graph
        self.assertEqual(12, len(all_intrinsics))

        # randomly check intrinsics from a few indices
        self.assertTrue(self.loader.get_camera_intrinsics(5).equals(all_intrinsics[5], 1e-5))
        self.assertTrue(self.loader.get_camera_intrinsics(7).equals(all_intrinsics[7], 1e-5))

    @patch("gtsfm.loader.loader_base.LoaderBase.is_valid_pair", return_value=True)
    def test_is_valid_pair_within_lookahead(self, base_is_valid_pair_mock: MagicMock) -> None:
        i1 = 1
        i2 = 3
        self.assertTrue(self.loader.is_valid_pair(i1, i2))
        base_is_valid_pair_mock.assert_called_once_with(i1, i2)

    @patch("gtsfm.loader.loader_base.LoaderBase.is_valid_pair", return_value=True)
    def test_is_valid_pair_outside_lookahead(self, base_is_valid_pair_mock: MagicMock) -> None:
        i1 = 1
        i2 = 10
        self.assertFalse(self.loader.is_valid_pair(i1, i2))
        base_is_valid_pair_mock.assert_called_once_with(i1, i2)


if __name__ == "__main__":
    unittest.main()
