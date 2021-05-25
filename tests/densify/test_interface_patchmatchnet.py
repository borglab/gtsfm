"""Unit tests for the interface for PatchmatchNet

Authors: Ren Liu
"""
import unittest
from pathlib import Path

import numpy as np
from gtsam import PinholeCameraCal3Bundler

from gtsfm.common.gtsfm_data import GtsfmData, SfmTrack
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.densify.interface_patchmatchnet import PatchmatchNetData

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"

DEFAULT_FOLDER = DATA_ROOT_PATH / "set1_lund_door"

NUM_VIEWS = 5


class TestInterfacePatchmatchNet(unittest.TestCase):
    """Unit tests for the interface for PatchmatchNet."""

    def setUp(self):
        """Set up the image dictionary and gtsfm result for the test."""
        super().setUp()

        # initialize Olsson Loader
        loader = OlssonLoader(str(DEFAULT_FOLDER), image_extension="JPG")
        self._num_images = len(loader)
        self._num_valid_cameras = self._num_images // 2
        valid_cameras = np.arange(self._num_valid_cameras)

        # form image dictionary
        self._img_dict = dict()
        for i in range(self._num_images):
            self._img_dict[i] = loader.get_image(i)

        self._sfm_result = GtsfmData(self._num_images)

        # add cameras to sfm result
        for i in valid_cameras:
            self._sfm_result.add_camera(
                i, PinholeCameraCal3Bundler(loader.get_camera_pose(i), loader.get_camera_intrinsics(i))
            )

        # add tracks to sfm result
        self._num_tracks = self._num_valid_cameras * 4
        for j in range(self._num_tracks):
            track_to_add = SfmTrack(np.array([0, -2.0, j]))
            for i in range(self._num_valid_cameras):
                track_to_add.add_measurement(idx=i, m=np.array([i, i]))
            self._sfm_result.add_track(track_to_add)

        self._min_depth = 0
        self._max_depth = self._num_tracks - 1

        self._dataset_patchmatchnet = PatchmatchNetData(self._img_dict, self._sfm_result, num_views=NUM_VIEWS)

    def test_initialize_and_configure(self):
        """test initialization method and whether configuration is correct"""

        self.assertEqual(len(self._dataset_patchmatchnet), self._num_valid_cameras)

    def test_get_item(self):
        """test get item method when yielding test data from dataset"""

        for _, sample in enumerate(self._dataset_patchmatchnet):
            B, C, H, W = sample["imgs"]["stage_0"].shape

            # test batch size
            self.assertEqual(B, NUM_VIEWS)

            # test images
            h, w, c = self._img_dict[0].value_array.shape
            self.assertEqual((H, W, C), (h, w, c))

            # test projection matrices
            sample_proj_mat = sample["proj_matrices"]["stage_0"][0]
            sample_camera = self._sfm_result.get_camera(0)
            self.assertAlmostEqual(
                np.sum(
                    np.abs(
                        sample_proj_mat[:3, :4]
                        - sample_camera.calibration().K() @ sample_camera.pose().inverse().matrix()[:3, :4]
                    )
                ),
                0.0,
            )

            # test depth range
            self.assertEqual(sample["depth_min"], self._min_depth)
            self.assertEqual(sample["depth_max"], self._max_depth)

            break


if __name__ == "__main__":
    unittest.main()
