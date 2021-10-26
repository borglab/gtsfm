"""Unit tests for using Patchmatchnet method for dense point cloud reconstruction.

In this unit test, we make a simple scenario that eight cameras are around a circle (radius = 40.0).
Every camera's pose is towards the center of the circle (0, 0, 0).

The input sparse point cloud includes 30 toy points uniformly located at the same line:
  0x + -2y + 1z = 0, i.e. x=0

Authors: Ren Liu
"""
import unittest

import numpy as np
from gtsam import Cal3_S2, PinholeCameraCal3_S2, Point3
from gtsam.examples import SFMdata

from gtsfm.common.image import Image
from gtsfm.common.gtsfm_data import GtsfmData, SfmTrack
from gtsfm.densify.mvs_patchmatchnet import MVSPatchmatchNet

# set the dummy image size as 400x300, with 3 channels
IMAGE_W = 400
IMAGE_H = 300
IMAGE_C = 3

# set dummy track points, the coordinates are in the world frame
NUM_TRACKS = 30
DUMMY_TRACK_PTS_WORLD = [Point3(0, 0.5 * float(i), float(i)) for i in range(NUM_TRACKS)]

# set dummy camera intrinsics
CAMERA_INTRINSICS = Cal3_S2(
    fx=100.0,
    fy=100.0,
    s=0.0,
    u0=IMAGE_W // 2,
    v0=IMAGE_H // 2,
)
# set dummy camera poses as described in GTSAM example
CAMERA_POSES = SFMdata.createPoses(CAMERA_INTRINSICS)
# set dummy camera instances
CAMERAS = [PinholeCameraCal3_S2(CAMERA_POSES[i], CAMERA_INTRINSICS) for i in range(len(CAMERA_POSES))]
NUM_CAMERAS = len(CAMERAS)
# the number of valid images should be equal to the number of cameras (with estimated pose)
NUM_IMAGES = NUM_CAMERAS

# build dummy image dictionary with dummy image shape
DUMMY_IMAGE_DICT = {i: Image(value_array=np.zeros([IMAGE_H, IMAGE_W, IMAGE_C], dtype=int)) for i in range(NUM_IMAGES)}


class TestMVSPatchmatchNet(unittest.TestCase):
    """Unit tests for PatchmatchNet method."""

    def setUp(self) -> None:
        """Set up the image dictionary and gtsfm result for the test."""
        super().setUp()

        # set the number of images
        self._num_images = NUM_IMAGES

        # set up the dummy image dictionary
        self._img_dict = DUMMY_IMAGE_DICT

        # initialize sfm result
        self._sfm_result = self.get_dummy_gtsfm_data()

        # use patchmatchnet to recontruct dense point cloud
        self._dense_points = MVSPatchmatchNet().densify(self._img_dict, self._sfm_result, max_num_views=NUM_IMAGES)

    def get_dummy_gtsfm_data(self) -> GtsfmData:
        """Create a new GtsfmData instance, add dummy cameras and tracks, and draw the track points on all images"""
        sfm_result = GtsfmData(self._num_images)
        # Assume all dummy cameras are valid cameras, add toy data for cameras
        for i in range(NUM_CAMERAS):
            sfm_result.add_camera(i, CAMERAS[i])

        # Calculate the measurements under each camera for all track points, then add toy data for tracks:
        for j in range(NUM_TRACKS):
            world_x = DUMMY_TRACK_PTS_WORLD[j]
            track_to_add = SfmTrack(world_x)

            for i in range(NUM_CAMERAS):
                u, v = sfm_result.get_camera(i).project(world_x)

                # color the track point with (r, g, b) = (255, 255, 255)
                self._img_dict[i].value_array[int(v + 0.5), int(u + 0.5), :] = 255

                track_to_add.add_measurement(idx=i, m=[u, v])
            sfm_result.add_track(track_to_add)

        return sfm_result

    def test_patchmatchnet_result(self) -> None:
        """Test whether the result point cloud calculated by PatchmatchNet correctly reflects the original shape"""

        # test the dense point cloud shape is correct
        n, d = self._dense_points.shape

        self.assertTrue(n > 0 and d == 3)

        x = self._dense_points[:, 0]
        y = self._dense_points[:, 1]
        z = self._dense_points[:, 2]

        # per the line `0x + -2y + 1z = 0`, so check the slope k_yz == 2 (avoid [y=0, z=0] case by adding an epsilon)
        eps = 1e-20
        mean_k_yz_error = np.abs((z - 0 + 2 * eps) / (y - 0 + eps)).mean() - 2

        # should be close to 0
        x_error = np.abs(x).mean()

        self.assertTrue(mean_k_yz_error < 0.2 and x_error < 0.5)


if __name__ == "__main__":
    unittest.main()
