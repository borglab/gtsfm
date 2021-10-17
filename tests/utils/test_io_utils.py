from pathlib import Path
from gtsfm.utils.io import extract_extrinsics_from_camera, recover_pose_from_extrinsics

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3

import gtsfm.utils.io as io_utils

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def test_load_image() -> None:
    """Ensure focal length can be read from EXIF, for an image w/ known EXIF."""
    img_fpath = TEST_DATA_ROOT / "set2_lund_door_nointrinsics/images/DSC_0001.JPG"
    img = io_utils.load_image(img_fpath)
    assert img.exif_data.get("FocalLength") == 29


def test_read_points_txt() -> None:
    """ """
    fpath = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output" / "points3D.txt"
    point_cloud, rgb = io_utils.read_points_txt(fpath)

    assert isinstance(point_cloud, np.ndarray)
    assert isinstance(rgb, np.ndarray)

    assert point_cloud.shape == (2122, 3)
    assert point_cloud.dtype == np.float64

    assert rgb.shape == (2122, 3)
    assert rgb.dtype == np.uint8


def test_read_points_txt_nonexistent_file() -> None:
    """Ensure that providing a path to a nonexistent file returns None for both return args."""
    fpath = "nonexistent_dir/points.txt"
    point_cloud, rgb = io_utils.read_points_txt(fpath)

    assert point_cloud is None
    assert rgb is None


def test_read_images_txt() -> None:
    """ """
    fpath = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output" / "images.txt"
    wTi_list, img_fnames = io_utils.read_images_txt(fpath)

    assert all([isinstance(wTi, Pose3) for wTi in wTi_list])
    assert all([isinstance(img_fname, str) for img_fname in img_fnames])

    expected_img_fnames = [
        "crane_mast_1.jpg",
        "crane_mast_2.jpg",
        "crane_mast_3.jpg",
        "crane_mast_4.jpg",
        "crane_mast_5.jpg",
        "crane_mast_6.jpg",
        "crane_mast_7.jpg",
        "crane_mast_8.jpg",
    ]
    assert img_fnames == expected_img_fnames


def test_read_images_txt_nonexistent_file() -> None:
    """Ensure that providing a path to a nonexistent file returns None for both return args."""
    fpath = "nonexistent_dir/images.txt"
    wTi_list, img_fnames = io_utils.read_images_txt(fpath)
    assert wTi_list is None
    assert img_fnames is None


def test_read_cameras_txt() -> None:
    """Ensure that shared calibration from COLMAP output is read in as a single calibration."""
    fpath = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output" / "cameras.txt"
    calibrations = io_utils.read_cameras_txt(fpath)

    assert isinstance(calibrations, list)
    assert all([isinstance(calibration, Cal3Bundler) for calibration in calibrations])

    assert len(calibrations) == 1
    K = calibrations[0]
    assert K.fx() == 2332.47
    assert K.px() == 2028
    assert K.py() == 1520


def test_read_cameras_txt_nonexistent_file() -> None:
    """Ensure that providing a path to a nonexistent file returns None for calibrations return arg."""
    fpath = "nonexistent_dir/cameras.txt"
    calibrations = io_utils.read_cameras_txt(fpath)
    assert calibrations is None


# TODO in future PR: add round-trip test on write poses to images.txt, and load poses (poses->extrinsics->poses)


def test_round_trip_images_txt() -> None:
    """Loads a pose from file that was produced after applying an alignment transformation to a camera frustum. Converts
    sample pose into extrinsics. Then converts these extrinsics back into a pose and verifies it matches
    with the original."""

    # Setup dummy pinhole camera object
    default_intrinsics = Cal3Bundler(fx=100, k1=0, k2=0, u0=0, v0=0)

    pose_from_file = np.load("gtsfm_incorrect_alignment_cameras_wTc.npy")
    firstPose = pose_from_file[0, :, :]
    wTc = Pose3(Rot3(firstPose[:3, :3]), firstPose[:3, 3])
    camera = PinholeCameraCal3Bundler(wTc, default_intrinsics)

    # pose -> extrinsics
    iRw_quaternion, itw = extract_extrinsics_from_camera(camera)

    # extrinsics -> recovered pose
    recovered_wTc = recover_pose_from_extrinsics(iRw_quaternion, itw)

    assert wTc == recovered_wTc


test_round_trip_images_txt()
