"""Unit tests for Tanks & Temple dataset loader.

Author: John Lambert
"""
import unittest
from pathlib import Path

import numpy as np
from gtsam import Cal3Bundler, Unit3

import gtsfm.utils.geometry_comparisons as geom_comp_utils
from gtsfm.common.image import Image
from gtsfm.loader.tanks_and_temples_loader import TanksAndTemplesLoader
from gtsfm.frontend.verifier.loransac import LoRansac


_TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "tanks_and_temples_barn"


class TanksAndTemplesLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        scene_name = "Barn"

        lidar_ply_fpath = None
        colmap_ply_fpath = None

        img_dir = _TEST_DATA_ROOT / scene_name
        poses_fpath = _TEST_DATA_ROOT / f"{scene_name}_COLMAP_SfM.log"
        ply_alignment_fpath = _TEST_DATA_ROOT / f"{scene_name}_trans.txt"
        bounding_polyhedron_json_fpath = _TEST_DATA_ROOT / f"{scene_name}.json"

        # Note: PLY files are not provided here, as they are too large to include as test data (300 MB each).
        self.loader = TanksAndTemplesLoader(
            img_dir=str(img_dir),
            poses_fpath=str(poses_fpath),
            bounding_polyhedron_json_fpath=str(bounding_polyhedron_json_fpath),
            ply_alignment_fpath=str(ply_alignment_fpath),
            lidar_ply_fpath=str(lidar_ply_fpath),
            colmap_ply_fpath=str(colmap_ply_fpath),
        )

    def test_get_camera_intrinsics_full_res(self) -> None:
        """Tests that expected camera intrinsics for zero'th image are returned."""
        intrinsics = self.loader.get_camera_intrinsics_full_res(index=0)
        assert isinstance(intrinsics, Cal3Bundler)

        # Should be half of image width (1920 / 2).
        assert np.isclose(intrinsics.px(), 960.0)

        # Should be half of image height (1080 / 2).
        assert np.isclose(intrinsics.py(), 540.0)

    def test_get_camera_pose(self) -> None:
        """Tests that expected GT camera pose for zero'th image are returned."""
        wTi = self.loader.get_camera_pose(index=0)

        det = np.linalg.det(wTi.rotation().matrix())
        assert np.isclose(det, 1.0)

        wRi = wTi.rotation().matrix()
        assert np.allclose(wRi @ wRi.T, np.eye(3))

        expected_wRi = np.array(
            [
                [-0.43322, -0.0555537, -0.899574],
                [0.0567814, 0.994434, -0.0887567],
                [0.899498, -0.0895302, -0.427654],
            ]
        )
        assert np.allclose(wTi.rotation().matrix(), expected_wRi)

        expected_wti = np.array([3.24711, 0.140327, 0.557239])
        assert np.allclose(wTi.translation(), expected_wti)

    def test_image_filenames(self) -> None:
        """Verify that image file names are provided correctly (used in NetVLAD)."""
        filenames = self.loader.image_filenames()

        expected_filenames = ["000001.jpg", "000002.jpg", "000003.jpg"]
        assert filenames == expected_filenames

    def test_get_image_fpath(self) -> None:
        """Tests that index 0 maps to image '0000001.jpg', which is the zero'th image in the Barn dataset."""
        fpath = self.loader.get_image_fpath(index=0)
        assert isinstance(fpath, Path)
        assert fpath.parts[-4:] == ("data", "tanks_and_temples_barn", "Barn", "000001.jpg")

    def test_get_image_full_res(self) -> None:
        """Verifies that the zero'th image has expected image dimensions."""
        image = self.loader.get_image_full_res(index=0)
        assert isinstance(image, Image)

        assert image.height == 1080
        assert image.width == 1920

    def test_synthetic_correspondences_have_zero_two_view_error(self) -> None:
        # Skip this test in the CI, and only uncomment it to run it locally, since it requires PLY.
        return
        # Compute 2-view error using a front-end.
        verifier = LoRansac(use_intrinsics_in_verification=True, estimation_threshold_px=0.5)

        i1 = 0
        i2 = 1

        keypoints_list, match_indices_dict = self.loader.generate_synthetic_correspondences(
            images=[], image_pairs=[(i1, i2)]
        )
        keypoints_i1, keypoints_i2 = keypoints_list

        wTi1 = self.loader.get_camera_pose(index=i1)
        wTi2 = self.loader.get_camera_pose(index=i2)

        i2Ti1 = wTi2.between(wTi1)
        i2Ri1_expected = i2Ti1.rotation()
        i2Ui1_expected = Unit3(i2Ti1.translation())

        camera_intrinsics_i1 = self.loader.get_camera_intrinsics_full_res(index=i1)
        camera_intrinsics_i2 = self.loader.get_camera_intrinsics_full_res(index=i2)

        i2Ri1_computed, i2Ui1_computed, verified_indices_computed, _ = verifier.verify(
            keypoints_i1,
            keypoints_i2,
            match_indices_dict[(i1, i2)],
            camera_intrinsics_i1,
            camera_intrinsics_i2,
        )

        rot_angular_err = geom_comp_utils.compute_relative_rotation_angle(i2Ri1_expected, i2Ri1_computed)
        direction_angular_err = geom_comp_utils.compute_relative_unit_translation_angle(i2Ui1_expected, i2Ui1_computed)

        print(f"Errors: rotation {rot_angular_err:.2f}, direction {direction_angular_err:.2f}")
        # if i2Ri1_expected is None:
        #     self.assertIsNone(i2Ri1_computed)
        # else:
        #     angular_err = geom_comp_utils.compute_relative_rotation_angle(i2Ri1_expected, i2Ri1_computed)
        #     self.assertLess(
        #         angular_err,
        #         ROTATION_ANGULAR_ERROR_DEG_THRESHOLD,
        #         msg=f"Angular error {angular_err:.1f} vs. tol. {ROTATION_ANGULAR_ERROR_DEG_THRESHOLD:.1f}",
        #     )
        # if i2Ui1_expected is None:
        #     self.assertIsNone(i2Ui1_computed)
        # else:
        #     self.assertLess(
        #         geom_comp_utils.compute_relative_unit_translation_angle(i2Ui1_expected, i2Ui1_computed),
        #         DIRECTION_ANGULAR_ERROR_DEG_THRESHOLD,
        #     )

