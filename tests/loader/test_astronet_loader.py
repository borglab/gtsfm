"""Unit tests for the COLMAP Loader class.

Authors: John Lambert
"""

import unittest
import random
from pathlib import Path

import numpy as np
import trimesh
from gtsam import Pose3, PinholeCameraCal3Bundler

from gtsfm.common.image import Image
from gtsfm.loader.astronet_loader import AstroNetLoader
import gtsfm.utils.io as io_utils

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
INDICES_TRACKS3D = np.random.randint(low = 0, high=99846, size=10)


class TestAstroNetLoader(unittest.TestCase):
    def setUp(self):
        """Set up the loader for the test."""
        super().setUp()

        data_dir = TEST_DATA_ROOT / "2011212_opnav_022"
        gt_scene_mesh_path = data_dir / "vesta_99846.ply"    

        self.loader = AstroNetLoader(
            data_dir,
            gt_scene_mesh_path=gt_scene_mesh_path,
            use_gt_intrinsics=True,
            use_gt_extrinsics=True,
            use_gt_tracks3D=True,
            max_frame_lookahead=2,
        )

    def test_constructor_set_properties(self) -> None:
        """Ensure that constructor sets class properties correctly."""
        assert self.loader._gt_scene_trimesh is not None
        assert self.loader._use_gt_intrinsics == True
        assert self.loader._use_gt_extrinsics == True
        assert self.loader._use_gt_tracks3D == True
        assert self.loader._max_frame_lookahead == 2

    def test_len(self) -> None:
        """Ensure we have one calibration per image/frame."""
        # there are 15 images and 99846 tracks in 2011212_opnav_022
        assert len(self.loader) == 15
        assert len(self.loader._calibrations) == 15
        assert self.loader._num_imgs == 15
        assert len(self.loader._image_paths) == 15

    def test_tracks3D(self) -> None:            
        """Ensure we have one track3D per mesh vertex"""
        # there are 15 images and 99846 tracks in 2011212_opnav_022
        assert self.loader._num_tracks3D == 99846
        assert len(self.loader._tracks3D) == 99846
        assert len(self.loader._gt_scene_trimesh.vertices) == 99846

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
            self.loader.get_image(len(self.loader))
        # index > len()
        with self.assertRaises(IndexError):
            self.loader.get_image(25)

    def test_get_camera_intrinsics(self) -> None:
        """Ensure that for shared calibration case, GT intrinsics are identical across frames."""
        # TODO: also test case with multiple calibrations
        K0 = self.loader.get_camera_intrinsics(0).K()
        K1 = self.loader.get_camera_intrinsics(1).K()

        # should be shared intrinsics
        np.testing.assert_allclose(K0, K1)

    def test_get_image(self) -> None:
        """Ensure a downsampled image can be successfully provided."""
        img0 = self.loader.get_image(0)
        assert isinstance(img0, Image)

    def test_get_track3D_valid_index(self):
        """Tests that get_track3D works for all valid indices."""
        for idx_track3D in INDICES_TRACKS3D:
            track3D = self.loader.get_track3D(idx_track3D)
            self.assertIsNotNone(track3D.point3())

    def test_get_track3D_invalid_index(self):
        """Test that get_track3D raises an exception on an invalid index."""
        # negative index
        with self.assertRaises(IndexError):
            self.loader.get_track3D(-1)
        # len() as index
        with self.assertRaises(IndexError):
            self.loader.get_track3D(self.loader._num_tracks3D)
        # index > len()
        with self.assertRaises(IndexError):
            self.loader.get_track3D(99999)

    def test_image_contents(self):
        """Test the actual image which is being fetched by the loader at an index.

        This test's primary purpose is to check if the ordering of filename is being respected by the loader
        """
        index_to_test = 5
        file_path = TEST_DATA_ROOT/ "2011212_opnav_022" / "images" / "00000005.png"
        loader_image = self.loader.get_image(index_to_test)
        expected_image = io_utils.load_image(file_path)
        np.testing.assert_allclose(expected_image.value_array, loader_image.value_array)

    def test_get_camera_pose_exists(self):
        """Tests that the correct pose is fetched (present on disk)."""
        fetched_pose = self.loader.get_camera_pose(1)
        self.assertTrue(isinstance(fetched_pose, Pose3))

    def test_get_camera_invalid_index(self):
        """Tests that the camera pose is None, because it is missing on disk."""
        with self.assertRaises(IndexError):
            fetched_pose = self.loader.get_camera_pose(25)
            self.assertIsNone(fetched_pose)

    def test_trimesh_tracks3D_align(self):
        """Tests that the ground truth scene mesh aligns with tracks3D"""
        prox = trimesh.proximity.ProximityQuery(self.loader._gt_scene_trimesh)
        tracks3D_points3D, surface_points3D = [], []
        for idx_track3D in INDICES_TRACKS3D: 
            point3D = self.loader.get_track3D(idx_track3D).point3().reshape((1, 3))               
            closest, _, _ = prox.on_surface(point3D)
            tracks3D_points3D.append(point3D)
            surface_points3D.append(closest)
        np.testing.assert_allclose(tracks3D_points3D, surface_points3D)

    def test_colmap2gtsfm(self):
        """Tests the colmap2gtsfm static method by forward projecting tracks.

        This test also verifys that all data was read and ordered correctly.
        """
        uvs_measured, uvs_expected = [], []
        for idx_track3D in INDICES_TRACKS3D:                
            track3D = self.loader.get_track3D(idx_track3D)
            for idx_meas in range(track3D.number_measurements()):
                image_id, uv_measured = track3D.measurement(idx_meas)
                cal3 = self.loader.get_camera_intrinsics(image_id)
                wTi = self.loader.get_camera_pose(image_id)
                cam = PinholeCameraCal3Bundler(wTi, cal3)
                uv_expected, _ = cam.projectSafe(track3D.point3())
                uvs_measured.append(uv_measured)
                uvs_expected.append(uv_expected)
        # assert all to within 1 pixel absolute difference
        np.testing.assert_allclose(uvs_measured, uvs_expected, atol=1)


if __name__ == "__main__":
    unittest.main()
