"""Unit tests for the AstrovisionLoader class.

Authors: Travis Driver
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from gtsam import Cal3_S2, PinholeCameraCal3_S2, Pose3  # type: ignore

import gtsfm.utils.io as io_utils
import thirdparty.colmap.scripts.python.read_write_model as colmap_io
from gtsfm.common.image import Image
from gtsfm.loader.astrovision_loader import AstrovisionLoader

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# These lists specify a subset of the Point3D IDs, and their corresponding SfmTrack index in the loader, for testing.
# Specifically, given some index `j`, the data in the Point3D with ID `TEST_POINT3D_IDS[j]` should match that in the
# SfmTrack indexed by `TEST_SFM_TRACKS_INDICES[j]`.
TEST_POINT3D_IDS = [7, 2560, 3063, 3252, 3424, 3534, 3653, 3786, 3920, 4062, 4420, 4698, 3311, 3963, 325]
TEST_SFM_TRACKS_INDICES = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]


class TestAstrovisionLoader(unittest.TestCase):
    """Class containing tests for the AstrovisionLoader."""

    def setUp(self):
        """Set up the loader for the test."""
        super().setUp()

        data_dir = TEST_DATA_ROOT / "astrovision" / "test_2011212_opnav_022"
        gt_scene_mesh_path = data_dir / "vesta_5002.ply"

        # Read in COLMAP-formatted data for comparison.
        self.cameras, self.images, self.points3d = colmap_io.read_model(str(data_dir))

        # Initialize Loader.
        self.loader = AstrovisionLoader(
            str(data_dir),
            gt_scene_mesh_path=str(gt_scene_mesh_path),
            use_gt_extrinsics=True,
            use_gt_sfm_tracks=True,
            max_frame_lookahead=2,
        )

    def test_constructor_set_properties(self) -> None:
        """Ensure that constructor sets class properties correctly."""
        assert self.loader._gt_scene_trimesh is not None
        assert self.loader._gt_scene_trimesh.vertices.shape[0] == 5002  # type: ignore
        assert self.loader._gt_scene_trimesh.faces.shape[0] == 10000  # type: ignore
        assert self.loader._use_gt_extrinsics
        assert self.loader._use_gt_sfm_tracks
        assert self.loader._max_frame_lookahead == 2

    def test_len(self) -> None:
        """Ensure we have one calibration per image/frame."""
        # there are 4 images and 1465 tracks in 2011212_opnav_022
        assert len(self.loader) == 4
        assert len(self.loader._calibrations) == 4
        assert self.loader._num_imgs == 4
        assert len(self.loader._image_paths) == 4

    def test_sfm_tracks(self) -> None:
        """Ensure we have one SfmTrack per mesh vertex"""
        # there are 4 images and 1465 tracks in 2011212_opnav_022
        assert self.loader.num_sfm_tracks == 1465
        assert self.loader._sfm_tracks is not None
        assert len(self.loader._sfm_tracks) == 1465

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
            self.loader.get_image(len(self.loader))
        # index > len()
        with self.assertRaises(IndexError):
            self.loader.get_image(25)

    def test_get_camera_intrinsics(self) -> None:
        """Ensure that for shared calibration case, GT intrinsics are identical across frames."""
        # TODO: also test case with multiple calibrations
        intrinsics0 = self.loader.get_camera_intrinsics(0)
        intrinsics1 = self.loader.get_camera_intrinsics(1)
        assert intrinsics0 is not None
        assert intrinsics1 is not None
        K0 = intrinsics0.K()
        K1 = intrinsics1.K()
        np.testing.assert_allclose(K0, K1)

    def test_get_image(self) -> None:
        """Ensure an image can be successfully provided."""
        img0 = self.loader.get_image(0)
        assert isinstance(img0, Image)

    def test_get_sfm_track_valid_index(self) -> None:
        """Tests that get_sfm_track works for all valid indices."""
        for index in TEST_SFM_TRACKS_INDICES:
            sfm_track = self.loader.get_sfm_track(index)
            assert sfm_track is not None
            self.assertIsNotNone(sfm_track.point3())

    def test_get_sfm_track_invalid_index(self) -> None:
        """Test that get_sfm_track raises an exception on an invalid index."""
        # negative index
        with self.assertRaises(IndexError):
            self.loader.get_sfm_track(-1)
        # len() as index
        with self.assertRaises(IndexError):
            self.loader.get_sfm_track(self.loader.num_sfm_tracks)
        # index > len()
        with self.assertRaises(IndexError):
            self.loader.get_sfm_track(99999)

    def test_image_contents(self) -> None:
        """Test the actual image which is being fetched by the loader at an index.

        This test's primary purpose is to check if the ordering of filename is being respected by the loader
        """
        index_to_test = 1
        file_path = TEST_DATA_ROOT / "astrovision" / "test_2011212_opnav_022" / "images" / "00000001.png"
        loader_image = self.loader.get_image(index_to_test)
        expected_image = io_utils.load_image(str(file_path))
        np.testing.assert_allclose(expected_image.value_array, loader_image.value_array)

    def test_get_camera_pose_exists(self) -> None:
        """Tests that the correct pose is fetched (present on disk)."""
        fetched_pose = self.loader.get_camera_pose(1)
        self.assertTrue(isinstance(fetched_pose, Pose3))

    def test_get_camera_invalid_index(self) -> None:
        """Tests that the camera pose is None, because it is missing on disk."""
        with self.assertRaises(IndexError):
            fetched_pose = self.loader.get_camera_pose(25)
            self.assertIsNone(fetched_pose)

    def test_sfm_tracks_point3(self) -> None:
        """Tests that the 3D point of the SfmTrack matches the AstroVision data."""
        sfm_tracks_point3s = [
            track.point3().flatten()
            for index in TEST_SFM_TRACKS_INDICES
            if (track := self.loader.get_sfm_track(index)) is not None
        ]
        astrovision_point3s = [self.points3d[point3d_id].xyz.flatten() for point3d_id in TEST_POINT3D_IDS]
        # np.testing.assert_allclose(sfm_tracks_point3s, self.loader._gt_scene_trimesh.vertices[TEST_POINT3D_IDS])
        np.testing.assert_allclose(sfm_tracks_point3s, astrovision_point3s)

    def test_sfm_tracks_measurements(self) -> None:
        """Tests that the SfmTrack measurements match the AstroVision data."""
        for idx in range(len(TEST_SFM_TRACKS_INDICES)):
            sfm_track = self.loader.get_sfm_track(TEST_SFM_TRACKS_INDICES[idx])
            assert sfm_track is not None
            sfm_track_measurements = [
                sfm_track.measurement(image_idx)[1].flatten()
                for image_idx in range(sfm_track.numberMeasurements())
                if sfm_track is not None
            ]
            point3d = self.points3d[TEST_POINT3D_IDS[idx]]
            astrovision_measurements = [
                self.images[image_id].xys[point2d_idx]
                for (image_id, point2d_idx) in zip(point3d.image_ids, point3d.point2D_idxs)
            ]
            np.testing.assert_allclose(sfm_track_measurements, astrovision_measurements)

    def test_colmap2gtsfm(self) -> None:
        """Tests the colmap2gtsfm static method by forward projecting tracks.

        This test also verifies that all data was read and ordered correctly.
        """
        uvs_measured, uvs_expected = [], []
        for index in TEST_SFM_TRACKS_INDICES:
            sfm_track = self.loader.get_sfm_track(index)
            assert sfm_track is not None
            for idx_meas in range(sfm_track.numberMeasurements()):
                image_id, uv_measured = sfm_track.measurement(idx_meas)
                cal3 = self.loader.get_camera_intrinsics(image_id)
                wTi = self.loader.get_camera_pose(image_id)
                assert isinstance(cal3, Cal3_S2)
                assert isinstance(wTi, Pose3)
                cam = PinholeCameraCal3_S2(wTi, cal3)
                uv_expected, _ = cam.projectSafe(sfm_track.point3())
                uvs_measured.append(uv_measured)
                uvs_expected.append(uv_expected)
        # assert all to within 1 pixel absolute difference
        np.testing.assert_allclose(uvs_measured, uvs_expected, atol=1)

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
        self.assertFalse(self.loader.is_valid_pair(i1, i2))
        base_is_valid_pair_mock.assert_called_once_with(i1, i2)


if __name__ == "__main__":
    unittest.main()
