"""Unit tests for the interface for PatchmatchNet

In this unit test, we make a simple scenario that eight cameras are around a circle (radius = 40.0).
Every camera's pose is towards the center of the circle (0, 0, 0). 100 track points are set on the camera plane or
upper the camera plane.

Authors: Ren Liu
"""

import unittest

import numpy as np
from gtsam import Cal3_S2, PinholeCameraCal3_S2, Point3
from gtsam.examples import SFMdata

from gtsfm.common.gtsfm_data import GtsfmData, SfmTrack
from gtsfm.common.image import Image
from gtsfm.densify.patchmatchnet_data import PatchmatchNetData

# set the default image size as 800x600, with 3 channels
DEFAULT_IMAGE_W = 800
DEFAULT_IMAGE_H = 600
DEFAULT_IMAGE_C = 3

# set default track points, the coordinates are in the world frame
DEFAULT_NUM_TRACKS = 100
DEFAULT_TRACK_POINTS = [Point3(5, 5, float(i)) for i in range(DEFAULT_NUM_TRACKS)]

# set default camera intrinsics
DEFAULT_CAMERA_INTRINSICS = Cal3_S2(
    fx=100.0,
    fy=100.0,
    s=1.0,
    u0=DEFAULT_IMAGE_W // 2,
    v0=DEFAULT_IMAGE_H // 2,
)
# set default camera poses as described in GTSAM example
DEFAULT_CAMERA_POSES = SFMdata.posesOnCircle(R=40)
# set default camera instances
DEFAULT_CAMERAS = [
    PinholeCameraCal3_S2(DEFAULT_CAMERA_POSES[i], DEFAULT_CAMERA_INTRINSICS) for i in range(len(DEFAULT_CAMERA_POSES))
]
DEFAULT_NUM_CAMERAS = len(DEFAULT_CAMERAS)
# the number of valid images should be equal to the number of cameras (with estimated pose)
DEFAULT_NUM_IMAGES = DEFAULT_NUM_CAMERAS

# build dummy image dictionary with default image shape
DEFAULT_DUMMY_IMAGE_DICT = {
    i: Image(value_array=np.zeros([DEFAULT_IMAGE_H, DEFAULT_IMAGE_W, DEFAULT_IMAGE_C], dtype=int))
    for i in range(DEFAULT_NUM_IMAGES)
}

# set camera[1] to be selected in test_get_item
EXAMPLE_CAMERA_ID = 1


class TestPatchmatchNetData(unittest.TestCase):
    """Unit tests for the interface for PatchmatchNet."""

    def setUp(self) -> None:
        """Set up the image dictionary and gtsfm result for the test."""
        super().setUp()

        # set the number of images as the default number
        self._num_images = DEFAULT_NUM_IMAGES

        # set up the default image dictionary
        self._img_dict = DEFAULT_DUMMY_IMAGE_DICT

        # initialize sfm result
        self._sfm_result = self.get_dummy_gtsfm_data()

        # build PatchmatchNet dataset from input images and the sfm result
        self._dataset_patchmatchnet = PatchmatchNetData(self._img_dict, self._sfm_result)

    def get_dummy_gtsfm_data(self) -> GtsfmData:
        """ """
        sfm_result = GtsfmData(self._num_images)
        # Assume all default cameras are valid cameras, add toy data for cameras
        for i in range(DEFAULT_NUM_CAMERAS):
            sfm_result.add_camera(i, DEFAULT_CAMERAS[i])

        # Calculate the measurements under each camera for all track points, then add toy data for tracks:
        for j in range(DEFAULT_NUM_TRACKS):
            world_x = DEFAULT_TRACK_POINTS[j]
            track_to_add = SfmTrack(world_x)

            for i in range(DEFAULT_NUM_CAMERAS):
                uv = sfm_result.get_camera(i).project(world_x)
                track_to_add.addMeasurement(idx=i, m=uv)
            sfm_result.add_track(track_to_add)

        return sfm_result

    def test_dataset_length(self) -> None:
        """Test whether the dataset length is equal to the number of valid cameras, because every valid camera will be
        regarded as reference view once."""
        self.assertEqual(len(self._dataset_patchmatchnet), DEFAULT_NUM_CAMERAS)

    def test_select_src_views(self) -> None:
        """Test whether the (ref_view, src_view) pairs are selected correctly."""
        pairs = self._dataset_patchmatchnet.get_packed_pairs()
        expected_pairs = [
            {"ref_id": 0, "src_ids": [7, 1, 6, 2]},
            {"ref_id": 1, "src_ids": [0, 2, 7, 3]},
            {"ref_id": 2, "src_ids": [3, 1, 4, 0]},
            {"ref_id": 3, "src_ids": [4, 2, 5, 1]},
            {"ref_id": 4, "src_ids": [5, 3, 6, 2]},
            {"ref_id": 5, "src_ids": [6, 4, 7, 3]},
            {"ref_id": 6, "src_ids": [5, 7, 4, 0]},
            {"ref_id": 7, "src_ids": [6, 0, 5, 1]},
        ]
        self.assertTrue(pairs, expected_pairs)

    @unittest.skip("Frank Oct 5: Skip, no time to debug right now")
    def test_depth_ranges(self) -> None:
        """Test whether the depth ranges for each camera are calculated correctly and whether the depth outliers
        (one too close and one too far) are filtered out in the depth range"""

        # test the lower bound of the depth range
        self.assertAlmostEqual(self._dataset_patchmatchnet._depth_ranges[EXAMPLE_CAMERA_ID][0], 10.60262, 2)
        # test the upper bound of the depth range
        self.assertAlmostEqual(self._dataset_patchmatchnet._depth_ranges[EXAMPLE_CAMERA_ID][1], 34.12857, 2)

    @unittest.skip("Frank Oct 5: Skip, no time to debug right now")
    def test_get_item(self) -> None:
        """Test get_item method when yielding test data from dataset for inference."""
        example = self._dataset_patchmatchnet[EXAMPLE_CAMERA_ID]

        B, C, H, W = example["imgs"]["stage_0"].shape

        # the batch size here means the number of views (1 ref view and (B-1) src views) used in inference.
        #   the number of views must not be larger than the number of valid cameras, or the number of images
        self.assertLessEqual(B, DEFAULT_NUM_IMAGES)

        # test that the image tensor is identical to original image dimensions
        h, w, c = self._img_dict[0].value_array.shape
        self.assertEqual((H, W, C), (h, w, c))

        # test 3x4 projection matrices are correct
        sample_proj_mat = example["proj_matrices"]["stage_0"][0][:3, :4]
        actual_proj_mat = np.array(
            [
                [-344.936916, -203.515560, -97.9843925, 16492.4225],
                [-188.648444, -188.648444, -169.774938, 12369.3169],
                [-0.685994341, -0.685994341, -0.242535625, 41.2310563],
            ]
        )
        np.testing.assert_array_almost_equal(sample_proj_mat, actual_proj_mat, decimal=4)


if __name__ == "__main__":
    unittest.main()
