"""Units tests for pyCOLMAP utilities.

Author: Travis Driver
"""
import unittest

import numpy as np
import gtsam

import gtsfm.utils.pycolmap_utils as pycolmap_utils  # this needs to be imported before
import thirdparty.colmap.scripts.python.read_write_model as colmap


def make_dummy_colmap_image(image_id: int, camera_id: int) -> colmap.Image:
    return colmap.Image(
        id=image_id,
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        camera_id=camera_id,
        name="dummy",
        xys=np.random.randint(0, 100, (5, 2)).astype(float),
        point3D_ids=np.array([0]),  # not currently used
    )


class TestPyColmapUtils(unittest.TestCase):
    """Class containing all unit tests for pyCOLMAP utils."""

    def test_point3d_to_sfmtrack_colmap_pinhole(self) -> None:
        """Test conversion of COLMAP's Point3D to SfmTrack with `PINHOLE` camera model."""
        point3d = colmap.Point3D(
            id=0,
            xyz=np.zeros(3),
            rgb=np.zeros(3),
            error=0,
            image_ids=np.array([0, 2, 7]),
            point2D_idxs=np.random.randint(0, 5, 3),
        )
        cameras = {0: colmap.Camera(id=0, model="PINHOLE", width=100, height=100, params=[1.0, 2.0, 3.0, 4.0])}
        images = {
            0: make_dummy_colmap_image(0, 0),
            2: make_dummy_colmap_image(2, 0),
            7: make_dummy_colmap_image(7, 0),
        }

        track, gtsfm_cameras = pycolmap_utils.point3d_to_sfmtrack(point3d, images, cameras)
        for meas in track.measurements:
            image_id = meas[0]
            point2d_idx = point3d.point2D_idxs[np.where(point3d.image_ids == image_id)[0]]
            assert np.linalg.norm(images[image_id].xys[point2d_idx] - meas[1]) == 0
            assert all([isinstance(camera, gtsam.PinholeCameraCal3_S2) for camera in gtsfm_cameras.values()])


if __name__ == "__main__":
    unittest.main()
