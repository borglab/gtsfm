
import unittest
from pathlib import Path

import numpy as np
from gtsam import Pose3

from gtsfm.common.image import Image
from gtsfm.loader.colmap_loader import ColmapLoader

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class TestColmapLoader(unittest.TestCase):

    def setUp(self):
        """Set up the loader for the test."""
        super().setUp()

        colmap_files_dirpath = TEST_DATA_ROOT / "set1_lund_door/colmap_ground_truth"
        images_dir = TEST_DATA_ROOT / "set1_lund_door/images"

        self.loader = ColmapLoader(
            colmap_files_dirpath,
            images_dir,
            use_gt_intrinsics= True,
            use_gt_extrinsics= True,
            max_frame_lookahead = 3,
            max_resolution = 500
        )

    def test_constructor_set_properties(self) -> None:
        """Ensure that constructor sets class properties correctly."""
        assert self.loader._use_gt_intrinsics == True
        assert self.loader._use_gt_extrinsics == True
        assert self.loader._max_frame_lookahead == 3
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
        # should be shared intrinsics
        np.testing.assert_allclose(self.loader.get_camera_intrinsics(0).K(), self.loader.get_camera_intrinsics(1).K())

    def test_image_resolution(self) -> None:
        """Ensure that the image is downsampled properly to a max resolution of 500 px.

        Note: native resolution is (1936, 1296) for (H,W)
        """
        assert self.loader._scale_u == 500 / 1296
        assert np.isclose(self.loader._scale_v, 500 / 1296, atol=1e-4)

        assert self.loader._target_h == 747
        assert self.loader._target_w == 500

        # ensure that the aspect ratios match up to 3 decimal places
        downsampled_aspect_ratio = self.loader._target_w / self.loader._target_h
        assert np.isclose(downsampled_aspect_ratio, 1296 / 1936, atol=1e-3)

    def test_get_image(self) -> None:
        """Ensure a downsampled image can be successfully provided."""
        img0 = self.loader.get_image(0)
        assert isinstance(img0, Image)

    def test_get_camera_pose(self) -> None:
        """Ensure a camera pose can be successfully provided"""
        wT0 = self.loader.get_camera_pose(0)
        assert isinstance(wT0, Pose3)


# TODO in future: instantiate an object while providing bad paths 
