"""Unit tests for the COLMAP Loader class.

Authors: John Lambert
"""

import unittest
from pathlib import Path

import numpy as np
from gtsam import Rot3, Pose3
from scipy.spatial.transform import Rotation

from gtsfm.common.image import Image
from gtsfm.loader.astronet_loader import AstroNetLoader

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class TestAstroNetLoader(unittest.TestCase):
    def setUp(self):
        """Set up the loader for the test."""
        super().setUp()

        data_dir = TEST_DATA_ROOT / "2011212_opnav_022/"

        self.loader = AstroNetLoader(
            data_dir,
            gt_scene_mesh_path=None,
            use_gt_intrinsics=True,
            use_gt_extrinsics=True,
            max_frame_lookahead=2,
        )

    def test_constructor_set_properties(self) -> None:
        """Ensure that constructor sets class properties correctly."""
        assert self.loader._gt_scene_trimesh is None
        assert self.loader._use_gt_intrinsics == True
        assert self.loader._use_gt_extrinsics == True
        assert self.loader._max_frame_lookahead == 2

    def test_len(self) -> None:
        """Ensure we have one calibration per image/frame."""
        # there are 15 images in 2011212_opnav_022
        assert len(self.loader) == 15
        assert len(self.loader._calibrations) == 15
        assert self.loader._num_imgs == 15
        assert len(self.loader._image_paths) == 15

    def test_get_camera_intrinsics(self) -> None:
        """Ensure that for shared calibration case, GT intrinsics are identical across frames."""
        K0 = self.loader.get_camera_intrinsics(0).K()
        K1 = self.loader.get_camera_intrinsics(1).K()

        # should be shared intrinsics
        np.testing.assert_allclose(K0, K1)

    def test_get_image(self) -> None:
        """Ensure a downsampled image can be successfully provided."""
        img0 = self.loader.get_image(0)
        assert isinstance(img0, Image)

    # def test_get_camera_pose(self) -> None:
    #     """Ensure a camera pose can be successfully provided"""
    #     wT0 = self.loader.get_camera_pose(0)
    #     assert isinstance(wT0, Pose3)

    #     # From images.txt files, for DSC_0001.JPG (0th image)
    #     qw, qx, qy, qz = 0.983789, 0.00113517, 0.176825, -0.0298644
    #     tx, ty, tz = -7.60712, 0.428157, 2.75243

    #     cRw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    #     ctw = np.array([tx, ty, tz])

    #     # COLMAP saves extrinsics as cTw, not poses wTc
    #     cTw_expected = Pose3(Rot3(cRw), ctw)
    #     wT0_expected = cTw_expected.inverse()
    #     np.testing.assert_allclose(wT0.rotation().matrix(), wT0_expected.rotation().matrix(), atol=1e-5)
    #     np.testing.assert_allclose(wT0.translation(), wT0_expected.translation(), atol=1e-5)

if __name__ == "__main__":
    unittest.main()
