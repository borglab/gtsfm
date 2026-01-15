"""
Unit tests for io utility functions.
Authors: Adi, Frank Dellaert.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, SfmTrack  # type: ignore

import gtsfm.utils.io as io_utils
import thirdparty.colmap.scripts.python.read_write_model as colmap_io
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class TestIoUtils(unittest.TestCase):
    def test_load_image(self) -> None:
        """Ensure focal length can be read from EXIF, for an image w/ known EXIF."""
        img_fpath = TEST_DATA_ROOT / "set2_lund_door_nointrinsics/images/DSC_0001.JPG"
        img = io_utils.load_image(img_fpath)
        assert img.exif_data is not None
        self.assertEqual(img.exif_data.get("FocalLength"), 29)

    def test_read_points_txt(self) -> None:
        """ """
        fpath = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output" / "points3D.txt"
        point_cloud, rgb = io_utils.read_points_txt(fpath)

        self.assertIsInstance(point_cloud, np.ndarray)
        self.assertIsInstance(rgb, np.ndarray)

        assert point_cloud is not None
        self.assertEqual(point_cloud.shape, (2122, 3))
        self.assertEqual(point_cloud.dtype, np.float64)

        assert rgb is not None
        self.assertEqual(rgb.shape, (2122, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_read_points_txt_nonexistent_file(self) -> None:
        """Ensure that providing a path to a nonexistent file returns None for both return args."""
        fpath = "nonexistent_dir/points.txt"
        point_cloud, rgb = io_utils.read_points_txt(fpath)

        self.assertIsNone(point_cloud)
        self.assertIsNone(rgb)

    def test_read_images_txt(self) -> None:
        """ """
        fpath = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output" / "images.txt"
        wTi_list, img_filenames = io_utils.read_images_txt(fpath)

        self.assertTrue(all([isinstance(wTi, Pose3) for wTi in wTi_list]))
        self.assertTrue(all([isinstance(img_fname, str) for img_fname in img_filenames]))

        expected_img_filenames = [
            "crane_mast_1.jpg",
            "crane_mast_2.jpg",
            "crane_mast_3.jpg",
            "crane_mast_4.jpg",
            "crane_mast_5.jpg",
            "crane_mast_6.jpg",
            "crane_mast_7.jpg",
            "crane_mast_8.jpg",
        ]
        self.assertEqual(img_filenames, expected_img_filenames)

    def test_read_images_txt_nonexistent_file(self) -> None:
        """Ensure that providing a path to a nonexistent file returns None for both return args."""
        fpath = "nonexistent_dir/images.txt"
        with self.assertRaises(FileNotFoundError):
            io_utils.read_images_txt(fpath)

    def test_read_cameras_txt(self) -> None:
        """Ensure that shared calibration from COLMAP output is read in as a single calibration."""
        fpath = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output" / "cameras.txt"
        calibrations, img_dims = io_utils.read_cameras_txt(fpath)

        assert calibrations is not None
        self.assertIsInstance(calibrations, list)
        self.assertTrue(all([isinstance(calibration, Cal3Bundler) for calibration in calibrations]))

        self.assertEqual(len(calibrations), 1)
        K = calibrations[0]
        self.assertEqual(K.fx(), 2332.47)
        self.assertEqual(K.px(), 2028)
        self.assertEqual(K.py(), 1520)
        self.assertEqual(K.k1(), 0.00400066)
        # COLMAP SIMPLE_RADIAL model has only 1 radial distortion coefficient.
        # A second radial distortion coefficient equal to zero is expected when it is converted to GTSAM's Cal3Bundler.
        self.assertEqual(K.k2(), 0)

        # Image dims is (H, W).
        assert img_dims is not None
        self.assertEqual(img_dims[0][0], 3040)
        self.assertEqual(img_dims[0][1], 4056)

    def test_read_cameras_txt_nonexistent_file(self) -> None:
        """Ensure that providing a path to a nonexistent file returns None for calibrations return arg."""
        fpath = "nonexistent_dir/cameras.txt"
        calibrations, img_dims = io_utils.read_cameras_txt(fpath)
        self.assertIsNone(calibrations)
        self.assertIsNone(img_dims)

    def test_round_trip_images_txt(self) -> None:
        """Verifies that round-trip saving and reading a COLMAP-style `images.txt` file yields input poses.

        Starts with a pose. Writes the pose to images.txt (in a temporary directory). Then reads images.txt to recover
        that same pose. Checks if the original wTc and recovered wTc match up.
        """
        # fmt: off
        # Rotation 45 degrees about the z-axis.
        original_wRc = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1]
            ]
        )
        original_wtc = np.array([3, -2, 1])
        # fmt: on

        # Setup dummy GtsfmData Object with one image
        original_wTc = Pose3(Rot3(original_wRc), original_wtc)
        default_intrinsics = Cal3Bundler(fx=100, k1=0, k2=0, u0=0, v0=0)
        camera = PinholeCameraCal3Bundler(original_wTc, default_intrinsics)
        gtsfm_data = GtsfmData(number_images=1)
        gtsfm_data.add_camera(0, camera)
        gtsfm_data.set_image_info(0, name="dummy_image.jpg", shape=(1, 1))

        # Perform write and read operations inside a temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            images_fpath = Path(tempdir) / "images.txt"
            gtsfm_data.write_images(tempdir)
            wTi_list, _ = io_utils.read_images_txt(images_fpath)
            recovered_wTc = wTi_list[0]

        npt.assert_almost_equal(original_wTc.matrix(), recovered_wTc.matrix(), decimal=3)

    def test_round_trip_cameras_txt(self) -> None:
        """Creates a two cameras and writes to cameras.txt (in a temporary directory). Then reads cameras.txt to recover
        the information. Checks if the original and recovered cameras match up."""

        # Create multiple calibration data
        k1 = Cal3Bundler(fx=100, k1=0, k2=0, u0=0, v0=0)
        k2 = Cal3Bundler(fx=200, k1=0.001, k2=0, u0=1000, v0=2000)
        k3 = Cal3Bundler(fx=300, k1=0.004, k2=0.001, u0=1001, v0=2002)
        original_calibrations = [k1, k2, k3]

        gtsfm_data = GtsfmData(number_images=len(original_calibrations))

        # Populate gtsfm_data with the generated vales
        for i in range(len(original_calibrations)):
            camera = PinholeCameraCal3Bundler(Pose3(), original_calibrations[i])
            gtsfm_data.add_camera(i, camera)

        # Create image infos
        for i in range(len(original_calibrations)):
            gtsfm_data.set_image_info(i, name=f"dummy_image_{i}.jpg", shape=(240, 320))

        # Round trip
        with tempfile.TemporaryDirectory() as tempdir:
            cameras_fpath = Path(tempdir) / "cameras.txt"

            gtsfm_data.write_cameras(tempdir)
            recovered_calibrations, _ = io_utils.read_cameras_txt(cameras_fpath)

        assert recovered_calibrations is not None
        self.assertEqual(len(original_calibrations), len(recovered_calibrations))

        for i in range(len(recovered_calibrations)):
            K_ori = original_calibrations[i]
            K_rec = recovered_calibrations[i]

            self.assertEqual(K_ori.fx(), K_rec.fx())
            self.assertEqual(K_ori.px(), K_rec.px())
            self.assertEqual(K_ori.py(), K_rec.py())
            self.assertEqual(K_ori.k1(), K_rec.k1())
            self.assertEqual(K_ori.k2(), K_rec.k2())

    def test_save_point_cloud_as_ply(self) -> None:
        """Round-trip test on .ply file read/write, with a point cloud colored as all red."""
        N = 10000
        # generate a cuboid of size 1 x 2 x 3 meters.
        points = np.random.uniform(low=[0, 0, 0], high=[1, 2, 3], size=(N, 3))
        # Color uniformly as red.
        rgb = np.zeros((N, 3), dtype=np.uint8)
        rgb[:, 0] = 255

        with tempfile.TemporaryDirectory() as tempdir:
            save_fpath = f"{tempdir}/pointcloud.ply"
            io_utils.save_point_cloud_as_ply(save_fpath=save_fpath, points=points, rgb=rgb)
            points_read, rgb_read = io_utils.read_point_cloud_from_ply(ply_fpath=save_fpath)

        np.testing.assert_allclose(points_read, points)
        np.testing.assert_allclose(rgb_read, rgb)
        self.assertEqual(rgb_read.dtype, np.uint8)

    def test_save_point_cloud_as_ply_uncolored(self) -> None:
        """Round-trip test on .ply file read/write, with an uncolored point cloud."""
        N = 10000
        points = np.random.uniform(low=[0, 0, 0], high=[1, 2, 3], size=(N, 3))

        with tempfile.TemporaryDirectory() as tempdir:
            save_fpath = f"{tempdir}/pointcloud.ply"
            io_utils.save_point_cloud_as_ply(save_fpath=save_fpath, points=points)
            points_read, rgb_read = io_utils.read_point_cloud_from_ply(ply_fpath=save_fpath)

        rgb_expected = np.zeros((N, 3), dtype=np.uint8)
        np.testing.assert_allclose(points, points_read)
        np.testing.assert_allclose(rgb_read, rgb_expected)
        self.assertEqual(rgb_read.dtype, np.uint8)

    def test_json_roundtrip(self) -> None:
        """Test that basic read/write to JSON works as intended."""
        data = {"data": [np.nan, -2.0, 999.0, 0.0]}
        with tempfile.TemporaryDirectory() as tempdir:
            json_fpath = f"{tempdir}/list_with_nan.json"
            io_utils.save_json_file(json_fpath=json_fpath, data=data)
            data_from_json = io_utils.read_json_file(fpath=json_fpath)

            # np.nan is output as null, then read in as None
            self.assertEqual(data_from_json["data"][0], None)
            np.testing.assert_allclose(data["data"][1:], data_from_json["data"][1:])


class TestColmapIO(unittest.TestCase):
    def test_sort_poses_and_filenames(self) -> None:
        """Tests that 5 image-camera pose pairs are sorted jointly according to file name."""
        wTi_list = [
            Pose3(Rot3(), np.array([0, 0, 34])),
            Pose3(Rot3(), np.array([0, 0, 35])),
            Pose3(Rot3(), np.array([0, 0, 36])),
            Pose3(Rot3(), np.array([0, 0, 28])),
            Pose3(Rot3(), np.array([0, 0, 37])),
        ]
        img_fnames = ["P1180334.JPG", "P1180335.JPG", "P1180336.JPG", "P1180328.JPG", "P1180337.JPG"]

        wTi_list_sorted, img_fnames_sorted, _ = io_utils.sort_poses_and_filenames(wTi_list, img_fnames)

        expected_img_fnames_sorted = ["P1180328.JPG", "P1180334.JPG", "P1180335.JPG", "P1180336.JPG", "P1180337.JPG"]
        self.assertEqual(img_fnames_sorted, expected_img_fnames_sorted)

        self.assertEqual(wTi_list_sorted[0].translation()[2], 28)
        self.assertEqual(wTi_list_sorted[1].translation()[2], 34)
        self.assertEqual(wTi_list_sorted[2].translation()[2], 35)
        self.assertEqual(wTi_list_sorted[3].translation()[2], 36)
        self.assertEqual(wTi_list_sorted[4].translation()[2], 37)

    def _check_scene_data(
        self, wTi_list, img_fnames, calibrations, point_cloud, rgb, img_dims, num_images, num_points
    ) -> None:
        """Common type and shape checks for COLMAP scene data."""
        self.assertTrue(all(isinstance(wTi, Pose3) for wTi in wTi_list))
        self.assertTrue(all(isinstance(fname, str) for fname in img_fnames))
        self.assertTrue(all(isinstance(cal, Cal3Bundler) for cal in calibrations))
        self.assertIsInstance(point_cloud, np.ndarray)
        self.assertIsInstance(rgb, np.ndarray)
        self.assertTrue(all(isinstance(dim, tuple) and len(dim) == 2 for dim in img_dims))

        self.assertEqual(len(wTi_list), num_images)
        self.assertEqual(len(img_fnames), num_images)
        self.assertEqual(len(calibrations), num_images)
        self.assertEqual(point_cloud.shape, (num_points, 3))
        self.assertEqual(rgb.shape, (num_points, 3))
        # Note: do not assert a specific first img dim; order may differ by dataset.

    def test_tracks_from_colmap(self) -> None:
        """Test conversion from COLMAP dicts to SfmTrack list."""
        data_dir = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output"
        _, images, points3d = colmap_io.read_model(path=str(data_dir), ext=".txt")
        sfm_tracks = io_utils.tracks_from_colmap(images, points3d)
        self.assertIsNotNone(sfm_tracks)
        assert sfm_tracks is not None
        self.assertTrue(all(isinstance(track, SfmTrack) for track in sfm_tracks))
        self.assertEqual(len(sfm_tracks), len(points3d))

    def test_colmap2gtsfm(self) -> None:
        """Test conversion from COLMAP dicts to GTSfM format."""
        data_dir = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output"
        cameras, images, points3d = colmap_io.read_model(path=str(data_dir), ext=".txt")
        img_fnames, wTi_list, calibrations, point_cloud, rgb, img_dims = io_utils.colmap2gtsfm(
            cameras, images, points3d
        )
        self._check_scene_data(wTi_list, img_fnames, calibrations, point_cloud, rgb, img_dims, 8, 2122)

    def test_read_scene_data_from_colmap_format(self) -> None:
        """Test reading a full COLMAP scene reconstruction model."""
        data_dir = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output"
        wTi_list, img_fnames, calibrations, point_cloud, rgb, img_dims = io_utils.read_scene_data_from_colmap_format(
            str(data_dir)
        )
        self._check_scene_data(wTi_list, img_fnames, calibrations, point_cloud, rgb, img_dims, 8, 2122)

    def test_tracks_from_colmap_unsorted(self) -> None:
        """Test conversion from COLMAP dicts to SfmTrack list for unsorted COLMAP data."""
        data_dir = TEST_DATA_ROOT / "unsorted_colmap"
        _, images, points3d = colmap_io.read_model(path=str(data_dir), ext=".txt")

        sfm_tracks = io_utils.tracks_from_colmap(images, points3d)
        self.assertIsNotNone(sfm_tracks)
        assert sfm_tracks is not None
        self.assertTrue(all(isinstance(track, SfmTrack) for track in sfm_tracks))
        self.assertEqual(len(sfm_tracks), len(points3d))

    def test_colmap2gtsfm_unsorted(self) -> None:
        """Test conversion from COLMAP dicts to GTSfM format for unsorted COLMAP data."""
        data_dir = TEST_DATA_ROOT / "unsorted_colmap"
        cameras, images, points3d = colmap_io.read_model(path=str(data_dir), ext=".txt")

        img_fnames = [img.name for img in images.values()]
        self.assertListEqual(img_fnames, ["bbb.jpg", "ccc.jpg", "aaa.jpg"])
        sorted_idx = io_utils._get_sorted_idx(img_fnames)
        self.assertListEqual(sorted_idx, [2, 0, 1])

        image_id_to_idx = io_utils.colmap_image_id_to_idx(images)
        self.assertDictEqual(image_id_to_idx, {300: 0, 100: 1, 200: 2})

        sorted_img_fnames, wTi_list, calibrations, point_cloud, rgb, img_dims = io_utils.colmap2gtsfm(
            cameras, images, points3d
        )
        self._check_scene_data(wTi_list, sorted_img_fnames, calibrations, point_cloud, rgb, img_dims, 3, 2)
        self.assertEqual(sorted_img_fnames, ["aaa.jpg", "bbb.jpg", "ccc.jpg"])
        self.assertEqual(img_dims, [(768, 1024), (3040, 4056), (600, 800)])

    def test_read_scene_data_from_colmap_format_unsorted(self) -> None:
        """Test reading a full COLMAP scene reconstruction model."""
        data_dir = TEST_DATA_ROOT / "unsorted_colmap"
        wTi_list, img_fnames, calibrations, point_cloud, rgb, img_dims = io_utils.read_scene_data_from_colmap_format(
            str(data_dir)
        )
        self._check_scene_data(wTi_list, img_fnames, calibrations, point_cloud, rgb, img_dims, 3, 2)
        self.assertEqual(img_fnames, ["aaa.jpg", "bbb.jpg", "ccc.jpg"])
        self.assertEqual(img_dims, [(768, 1024), (3040, 4056), (600, 800)])


if __name__ == "__main__":
    unittest.main()
