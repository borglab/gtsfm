"""Unit tests for the COLMAP Loader class.

Authors: John Lambert
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from gtsam import Pose3, Rot3, Cal3Bundler
from scipy.spatial.transform import Rotation

from gtsfm.common.image import Image
from gtsfm.loader.colmap_loader import ColmapLoader

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class TestColmapLoader(unittest.TestCase):
    def setUp(self):
        """Set up the loader for the test."""
        super().setUp()

        dataset_dir = TEST_DATA_ROOT / "set1_lund_door/colmap_ground_truth"
        images_dir = TEST_DATA_ROOT / "set1_lund_door/images"

        self.loader = ColmapLoader(
            dataset_dir,
            images_dir,
            use_gt_intrinsics=True,
            use_gt_extrinsics=True,
            max_resolution=500,
        )

    def test_constructor_set_properties(self) -> None:
        """Ensure that constructor sets class properties correctly."""
        assert self.loader._use_gt_intrinsics
        assert self.loader._use_gt_extrinsics
        assert self.loader._max_resolution == 500

    def test_len(self) -> None:
        """Ensure we have one calibration per image/frame."""
        # there are 12 images in Lund Door set 1
        assert len(self.loader) == 12
        assert len(self.loader._calibrations) == 12
        assert self.loader._num_imgs == 12
        assert len(self.loader._image_paths) == 12

    def test_get_camera_intrinsics(self) -> None:
        """Ensure that for shared calibration case, GT intrinsics are identical across frames."""
        K0 = self.loader.get_camera_intrinsics(0).K()
        K1 = self.loader.get_camera_intrinsics(1).K()
        # should be shared intrinsics
        np.testing.assert_allclose(K0, K1)

        # COLMAP estimates that original camera parameters are:
        w_orig = 1296
        h_orig = 1936
        f_orig = 2435.38
        px_orig = 648
        py_orig = 968

        scale_u = 500.0 / w_orig
        scale_v = 747.0 / h_orig

        f = K0[0, 0]
        px = K0[0, 2]
        py = K0[1, 2]

        assert np.isclose(f, f_orig * scale_u)
        assert np.isclose(px, px_orig * scale_u)
        assert np.isclose(py, py_orig * scale_v)

    def test_image_resolution(self) -> None:
        """Ensure that the image is downsampled properly to a max resolution of 500 px.

        Note: full resolution is (1936, 1296) for (H,W)
        """
        img0_full_res = self.loader.get_image_full_res(0)
        assert img0_full_res.width == 1296
        assert img0_full_res.height == 1936

        img0 = self.loader.get_image(0)

        # scale_u is 500.0 / 1296.0
        # scale_v is 500.0 / 1296.0
        # downsampled aspect ratio is 1296 / 1936
        assert img0.width == 500
        assert img0.height == 747

    def test_get_image(self) -> None:
        """Ensure a downsampled image can be successfully provided."""
        img0 = self.loader.get_image(0)
        assert isinstance(img0, Image)

    def test_get_camera_pose(self) -> None:
        """Ensure a camera pose can be successfully provided"""
        wT0 = self.loader.get_camera_pose(0)
        assert isinstance(wT0, Pose3)

        # From images.txt files, for DSC_0001.JPG (0th image)
        qw, qx, qy, qz = 0.983789, 0.00113517, 0.176825, -0.0298644
        tx, ty, tz = -7.60712, 0.428157, 2.75243

        cRw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        ctw = np.array([tx, ty, tz])

        # COLMAP saves extrinsics as cTw, not poses wTc
        cTw_expected = Pose3(Rot3(cRw), ctw)
        wT0_expected = cTw_expected.inverse()
        np.testing.assert_allclose(wT0.rotation().matrix(), wT0_expected.rotation().matrix(), atol=1e-5)
        np.testing.assert_allclose(wT0.translation(), wT0_expected.translation(), atol=1e-5)

    @patch("gtsfm.loader.loader_base.LoaderBase.is_valid_pair", return_value=True)
    def test_is_valid_pair_within_lookahead(self, base_is_valid_pair_mock: MagicMock) -> None:
        i1 = 5
        i2 = 7
        self.assertTrue(self.loader.is_valid_pair(i1, i2))
        base_is_valid_pair_mock.assert_called_once_with(i1, i2)

    @patch("gtsfm.loader.loader_base.LoaderBase.is_valid_pair", return_value=True)
    def test_is_valid_pair_outside_lookahead(self, base_is_valid_pair_mock: MagicMock) -> None:
        i1 = 5
        i2 = 15
        # Max frame lookahead is determined by retriever, not by loader, so this should be a valid pair.
        self.assertTrue(self.loader.is_valid_pair(i1, i2))
        base_is_valid_pair_mock.assert_called_once_with(i1, i2)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("gtsfm.loader.colmap_loader.io_utils.read_scene_data_from_colmap_format")
    def test_colmap_files_subdir(self, read_scene_data_mock: MagicMock, _: MagicMock) -> None:
        """Ensure COLMAP metadata can be loaded from a subdirectory under dataset_dir."""
        dataset_dir = TEST_DATA_ROOT / "set1_lund_door"
        colmap_files_subdir = "colmap_ground_truth"
        read_scene_data_mock.return_value = (
            [Pose3()],
            ["DSC_0001.JPG"],
            [Cal3Bundler()],
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            [],
        )

        loader = ColmapLoader(
            dataset_dir=str(dataset_dir),
            colmap_files_subdir=colmap_files_subdir,
        )

        expected_colmap_dir = str(dataset_dir / colmap_files_subdir)
        read_scene_data_mock.assert_called_once_with(expected_colmap_dir)
        assert loader._images_dir == str(dataset_dir / "images")


# TODO in future: instantiate an object while providing bad paths
if __name__ == "__main__":
    unittest.main()
