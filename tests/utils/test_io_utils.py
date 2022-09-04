import os
import tempfile
import unittest
from pathlib import Path

import gtsam
import numpy as np
import numpy.testing as npt
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3

import gtsfm.utils.io as io_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class TestIoUtils(unittest.TestCase):
    def test_load_image(self) -> None:
        """Ensure focal length can be read from EXIF, for an image w/ known EXIF."""
        img_fpath = TEST_DATA_ROOT / "set2_lund_door_nointrinsics/images/DSC_0001.JPG"
        img = io_utils.load_image(img_fpath)
        self.assertEqual(img.exif_data.get("FocalLength"), 29)

    def test_read_points_txt(self) -> None:
        """ """
        fpath = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output" / "points3D.txt"
        point_cloud, rgb = io_utils.read_points_txt(fpath)

        self.assertIsInstance(point_cloud, np.ndarray)
        self.assertIsInstance(rgb, np.ndarray)

        self.assertEqual(point_cloud.shape, (2122, 3))
        self.assertEqual(point_cloud.dtype, np.float64)

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
        wTi_list, img_filenames = io_utils.read_images_txt(fpath)
        self.assertIsNone(wTi_list)
        self.assertIsNone(img_filenames)

    def test_read_cameras_txt(self) -> None:
        """Ensure that shared calibration from COLMAP output is read in as a single calibration."""
        fpath = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output" / "cameras.txt"
        calibrations = io_utils.read_cameras_txt(fpath)

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

    def test_read_cameras_txt_nonexistent_file(self) -> None:
        """Ensure that providing a path to a nonexistent file returns None for calibrations return arg."""
        fpath = "nonexistent_dir/cameras.txt"
        calibrations = io_utils.read_cameras_txt(fpath)
        self.assertIsNone(calibrations)

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
        original_wtc = np.array([3,-2,1])
        # fmt: on

        # Setup dummy GtsfmData Object with one image
        original_wTc = Pose3(Rot3(original_wRc), original_wtc)
        default_intrinsics = Cal3Bundler(fx=100, k1=0, k2=0, u0=0, v0=0)
        camera = PinholeCameraCal3Bundler(original_wTc, default_intrinsics)
        gtsfm_data = GtsfmData(number_images=1)
        gtsfm_data.add_camera(0, camera)

        image = Image(value_array=None, file_name="dummy_image.jpg")
        images = [image]

        # Perform write and read operations inside a temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            images_fpath = os.path.join(tempdir, "images.txt")

            io_utils.write_images(gtsfm_data, images, tempdir)
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

        # Generate dummy images
        image = Image(value_array=np.zeros((240, 320)), file_name="dummy_image.jpg")
        images = [image for i in range(len(original_calibrations))]

        # Round trip
        with tempfile.TemporaryDirectory() as tempdir:
            cameras_fpath = os.path.join(tempdir, "cameras.txt")

            io_utils.write_cameras(gtsfm_data, images, tempdir)
            recovered_calibrations = io_utils.read_cameras_txt(cameras_fpath)

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

    def test_read_bal(self) -> None:
        """Check that read_bal creates correct GtsfmData object."""
        filename: str = gtsam.findExampleDataFile("5pointExample1.txt")
        data: GtsfmData = io_utils.read_bal(filename)
        self.assertEqual(data.number_images(), 2)
        self.assertEqual(data.number_tracks(), 5)

    def test_read_bundler(self) -> None:
        """Check that read_bal creates correct GtsfmData object."""
        filename: str = gtsam.findExampleDataFile("Balbianello.out")
        data: GtsfmData = io_utils.read_bundler(filename)
        self.assertEqual(data.number_images(), 5)
        self.assertEqual(data.number_tracks(), 544)

    def test_json_roundtrip(self) -> None:
        """Test that basic read/write to JSON works as intended."""
        data = {"data": [np.NaN, -2.0, 999.0, 0.0]}
        with tempfile.TemporaryDirectory() as tempdir:
            json_fpath = f"{tempdir}/list_with_nan.json"
            io_utils.save_json_file(json_fpath=json_fpath, data=data)
            data_from_json = io_utils.read_json_file(fpath=json_fpath)

            # np.NaN is output as null, then read in as None
            self.assertEqual(data_from_json["data"][0], None)
            np.testing.assert_allclose(data["data"][1:], data_from_json["data"][1:])


if __name__ == "__main__":
    unittest.main()
