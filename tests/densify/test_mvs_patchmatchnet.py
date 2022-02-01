"""Unit tests for using Patchmatchnet method for dense point cloud reconstruction.

In this unit test, we make a simple scenario that eight cameras are around a circle (radius = 40.0).
Every camera's pose is towards the center of the circle (0, 0, 0).

The input sparse point cloud includes 30 toy points uniformly located at the same line:
  0x + -2y + 1z = 0, i.e. x=0

Authors: Ren Liu
"""
from curses import noecho
import unittest

import numpy as np
import torch
from gtsam import Cal3_S2, PinholeCameraCal3_S2, Point3
from gtsam.examples import SFMdata

from gtsfm.common.image import Image
from gtsfm.common.gtsfm_data import GtsfmData, SfmTrack
from gtsfm.densify.mvs_patchmatchnet import MVSPatchmatchNet
from gtsfm.densify.patchmatchnet_data import PatchmatchNetData


# set dummy random seed
torch.manual_seed(0)

# set the dummy image size as 400x300, with 3 channels
IMAGE_W = 400
IMAGE_H = 300
IMAGE_C = 3

# set dummy track points, the coordinates are in the world frame
NUM_TRACKS = 30
DUMMY_TRACK_PTS_WORLD = [Point3(0, 0.5 * float(i), float(i)) for i in range(NUM_TRACKS)]

# dummy cameras are around a circle of radius 40.0
CAMERA_CIRCLE_RADIUS = 40.0
CAMERA_HEIGHT = 10.0
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

# a reconstructed point is consistent in geometry if it satisfies all geometric thresholds in more than 3 source views
MIN_NUM_CONSISTENT_VIEWS = 3
# the reprojection error in pixel coordinates should be less than 1
MAX_GEOMETRIC_PIXEL_THRESH = 1


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

        # Use patchmatchnet to recontruct dense point cloud. Discarding per-point colors.
        self._dense_points, _, _ = MVSPatchmatchNet().densify(
            images=self._img_dict,
            sfm_result=self._sfm_result,
            max_num_views=NUM_IMAGES,
            min_num_consistent_views=MIN_NUM_CONSISTENT_VIEWS,
        )

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
                self._img_dict[i].value_array[np.round(v).astype(np.uint32), np.round(u).astype(np.uint32), :] = 255

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

    def test_compute_filtered_reprojection_error(self) -> None:
        """Test whether the compute_filtered_reprojection_error produces correct reprojection errors
        In this test, we assume there is an object O at (0, 0, 0), where all cameras look at.
        Then we select camera 0 as dummy reference view, while camera 1 as dummy source view.
        """
        # prepare PatchmatchNet dataset
        dataset = PatchmatchNetData(images=self._img_dict, sfm_result=self._sfm_result, max_num_views=NUM_IMAGES)

        # set dummy reference and source view pair
        dummy_ref_view = 0
        dummy_src_view = 1

        # fetch depthmap resolution
        height, width = self._img_dict[dummy_ref_view].value_array.shape[:2]

        # set dummy estimated depthmap for reference and source view
        dummy_ref_depthmap = np.zeros([1, height, width])
        dummy_ref_depthmap[0, height // 2, width // 2] = np.linalg.norm([CAMERA_CIRCLE_RADIUS, CAMERA_HEIGHT])

        dummy_src_depthmap = np.zeros([1, height, width])
        # add some error when estimating the depthmap of source view
        dummy_src_depthmap[0, height // 2, width // 2] = np.linalg.norm([CAMERA_CIRCLE_RADIUS, CAMERA_HEIGHT]) + 0.5

        # build depthmap list
        depth_list = {dummy_ref_view: dummy_ref_depthmap, dummy_src_view: dummy_src_depthmap}

        # set dummy joint mask to only include the image center, where object O is located
        dummy_joint_mask = np.zeros([height, width], dtype=np.bool)
        dummy_joint_mask[height // 2, width // 2] = True

        # calculate reprojection error
        reproject_errors = MVSPatchmatchNet().compute_filtered_reprojection_error(
            dataset=dataset,
            ref_view=dummy_ref_view,
            src_views=[dummy_src_view],
            depth_list=depth_list,
            max_reprojection_err=MAX_GEOMETRIC_PIXEL_THRESH,
            joint_mask=dummy_joint_mask,
        )

        self.assertAlmostEqual(reproject_errors[0], 0.828, 2)


if __name__ == "__main__":
    unittest.main()
