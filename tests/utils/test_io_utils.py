from pathlib import Path

import numpy as np
from gtsam import Cal3Bundler, Pose3

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
