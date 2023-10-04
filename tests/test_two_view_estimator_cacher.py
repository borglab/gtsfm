"""Unit tests for two-view estimator cacher.

"""

import unittest
from unittest.mock import MagicMock, patch

from gtsfm.two_view_estimator_cacher import TwoViewEstimatorCacher


class TestTwoViewEstimatorCacher(unittest.TestCase):
    def setUp(self) -> None:

        self.keypoints_i1 = Keypoints(normalized_coordinates_i1)
        self.keypoints_i2 = Keypoints(normalized_coordinates_i2)
        self.corr_idxs = np.hstack([np.arange(5).reshape(-1, 1)] * 2)

        self.dummy_output = (None, None, None, None, None, None)

    @patch("gtsfm.utils.cache.generate_hash_for_image", return_value="img_key")
    @patch("gtsfm.utils.io.read_from_bz2_file", return_value=None)
    @patch("gtsfm.utils.io.write_to_bz2_file")
    def test_cache_miss(
        self, write_mock: MagicMock, read_mock: MagicMock, generate_hash_for_image_mock: MagicMock
    ) -> None:
        """Test the scenario of cache miss."""

        # Mock the underlying two-view estimator which is used on cache miss.
        underlying_estimator_mock = MagicMock()
        underlying_estimator_mock.run_2view.return_value = self.dummy_output
        underlying_estimator_mock.__class__.__name__ = "mock_two_view_estimator"

        cacher = TwoViewEstimatorCacher(two_view_estimator_obj=underlying_estimator_mock)

        result = obj_under_test.run_2view(
            keypoints_i1=self.keypoints_i1,
            keypoints_i2=self.keypoints_i2,
            putative_corr_idxs=self.corr_idxs,
            # camera_intrinsics_i1: Optional[gtsfm_types.CALIBRATION_TYPE],
            # camera_intrinsics_i2: Optional[gtsfm_types.CALIBRATION_TYPE],
            i2Ti1_prior=None,
            gt_camera_i1=None,
            gt_camera_i2=None,
            gt_scene_mesh=None,
        )
        # Assert the returned value.
        self.assertEqual(computed_keypoints, DUMMY_KEYPOINTS)
        np.testing.assert_allclose(computed_descriptors, DUMMY_DESCRIPTORS)

        # Assert that underlying object was called.
        underlying_detector_descriptor_mock.detect_and_describe.assert_called_once_with(DUMMY_IMAGE)

        # Assert that hash generation was called with the input image.
        generate_hash_for_image_mock.assert_called_with(DUMMY_IMAGE)

        # Assert that read function was called once and write function was called once.
        cache_path = ROOT_PATH / "cache" / "two_view_estimator" / "mock_two_view_estimator_img_key.pbz2"
        read_mock.assert_called_once_with(cache_path)
        write_mock.assert_called_once_with({"keypoints": DUMMY_KEYPOINTS, "descriptors": DUMMY_DESCRIPTORS}, cache_path)

    @patch("gtsfm.utils.cache.generate_hash_for_image", return_value="img_key")
    @patch(
        "gtsfm.utils.io.read_from_bz2_file",
        return_value={"keypoints": DUMMY_KEYPOINTS, "descriptors": DUMMY_DESCRIPTORS},
    )
    @patch("gtsfm.utils.io.write_to_bz2_file")
    def test_cache_hit(self, write_mock: MagicMock, read_mock: MagicMock, generate_hash_for_image_mock: MagicMock):
        """Test the scenario of cache miss."""

        # Mock the underlying two-view estimator which is used on cache miss.
        underlying_estimator_mock = MagicMock()
        # underlying_estimator_mock.run_2view.return_value = self.dummy_output
        underlying_estimator_mock.__class__.__name__ = "mock_two_view_estimator"

        cacher = TwoViewEstimatorCacher(two_view_estimator_obj=underlying_estimator_mock)

        result = obj_under_test.run_2view(
            keypoints_i1=self.keypoints_i1,
            keypoints_i2=self.keypoints_i2,
            putative_corr_idxs=self.corr_idxs,
            # camera_intrinsics_i1: Optional[gtsfm_types.CALIBRATION_TYPE],
            # camera_intrinsics_i2: Optional[gtsfm_types.CALIBRATION_TYPE],
            i2Ti1_prior=None,
            gt_camera_i1=None,
            gt_camera_i2=None,
            gt_scene_mesh=None,
        )

        # Assert the returned value.
        self.assertEqual(computed_keypoints, DUMMY_KEYPOINTS)
        np.testing.assert_allclose(computed_descriptors, DUMMY_DESCRIPTORS)

        # Assert that underlying object was not called.
        underlying_detector_descriptor_mock.detect_and_describe.assert_not_called()

        # Assert that hash generation was called with the input image.
        generate_hash_for_image_mock.assert_called_with(DUMMY_IMAGE)

        # Assert that read function was called once and write function was called once.
        cache_path = ROOT_PATH / "cache" / "two_view_estimator" / "mock_two_view_estimator_img_key.pbz2"
        read_mock.assert_called_once_with(cache_path)
        write_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
