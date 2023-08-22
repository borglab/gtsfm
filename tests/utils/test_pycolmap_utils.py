"""Units tests for pyCOLMAP utilities.

Author: Travis Driver
"""
import unittest
import os
from pathlib import Path

import numpy as np
import gtsam
import pycolmap

import gtsfm.utils.pycolmap_utils as pycolmap_utils  # this needs to be imported before
import thirdparty.colmap.scripts.python.read_write_model as colmap


TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class TestPyColmapUtils(unittest.TestCase):
    """Class containing all unit tests for pyCOLMAP utils."""

    def test_point3d_to_sfmtrack_colmap_pinhole(self) -> None:
        """Test conversion of COLMAP's Point3D to SfmTrack with `PINHOLE` camera model."""
        cameras, images, points3d = colmap.read_model(
            os.path.join(TEST_DATA_ROOT, "astrovision", "test_2011212_opnav_022")
        )
        point3d = points3d[14]  # chosen because it is seen in all images
        track, gtsfm_cameras = pycolmap_utils.point3d_to_sfmtrack(point3d, images, cameras)
        for meas in track.measurements:
            image_id = meas[0]
            point2d_idx = point3d.point2D_idxs[np.where(point3d.image_ids == image_id)[0]]
            assert np.linalg.norm(images[image_id].xys[point2d_idx] - meas[1]) == 0
            assert all([isinstance(camera, gtsam.PinholeCameraCal3_S2) for camera in gtsfm_cameras.values()])

    def test_point3d_to_sfmtrack_pycolmap_pinhole(self) -> None:
        """Test conversion of COLMAP's Point3D to SfmTrack with `PINHOLE` camera model."""
        recon = pycolmap.Reconstruction(os.path.join(TEST_DATA_ROOT, "astrovision", "test_2011212_opnav_022"))
        point3d = recon.points3D[14]  # chosen because it is seen in all images
        track, gtsfm_cameras = pycolmap_utils.point3d_to_sfmtrack(point3d, recon.images, recon.cameras)
        point3d_image_ids = np.array([ele.image_id for ele in point3d.track.elements])
        print(point3d_image_ids)
        for meas in track.measurements:
            image_id = int(meas[0])
            point2d_idx = point3d.track.elements[np.where(point3d_image_ids == image_id)[0][0]].point2D_idx
            assert np.linalg.norm(recon.images[image_id].points2D[point2d_idx].xy - meas[1]) == 0
            assert all([isinstance(camera, gtsam.PinholeCameraCal3_S2) for camera in gtsfm_cameras.values()])


if __name__ == "__main__":
    unittest.main()
