"""Unit tests for two-view estimator cacher."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from gtsam import Cal3Bundler

from gtsfm.common.keypoints import Keypoints
from gtsfm.products.two_view_result import TwoViewResult
from gtsfm.two_view_estimator_cacher import TwoViewEstimatorCacher

ROOT_PATH = Path(__file__).resolve().parent.parent

_DUMMY_OUTPUT = TwoViewResult(None, None, np.array([]), None, None, None)


class TestTwoViewEstimatorCacher(unittest.TestCase):
    def setUp(self) -> None:
        # Generate 20 random keypoints, and assume H = W = 1000.
        coordinates_i1 = np.random.randint(low=0, high=1000, size=(20, 2))
        coordinates_i2 = np.random.randint(low=0, high=1000, size=(20, 2))

        self.keypoints_i1 = Keypoints(coordinates_i1)
        self.keypoints_i2 = Keypoints(coordinates_i2)
        # Horizontally stack two (5,1) arrays to form (20,2).
        self.corr_idxs = np.hstack([np.arange(5).reshape(-1, 1)] * 2)

        fx, k1, k2, u0, v0 = 583.1175, 0, 0, 507, 380
        calibration = Cal3Bundler(fx, k1, k2, u0, v0)
        self.camera_intrinsics_i1 = calibration
        self.camera_intrinsics_i2 = calibration

        self.dummy_output = _DUMMY_OUTPUT

    @patch("gtsfm.utils.cache.generate_hash_for_numpy_array", return_value="numpy_key")
    @patch("gtsfm.utils.io.read_from_bz2_file", return_value=None)
    @patch("gtsfm.utils.io.write_to_bz2_file")
    def test_cache_miss(
        self, write_mock: MagicMock, read_mock: MagicMock, generate_hash_for_numpy_array_mock: MagicMock
    ) -> None:
        """Test the scenario of cache miss for the TwoViewEstimator."""

        # Mock the underlying two-view estimator which is used on cache miss.
        underlying_estimator_mock = MagicMock()
        underlying_estimator_mock.run_2view.return_value = self.dummy_output
        verifier_key = "Ransac__use_intrinsicsTrue_4px"

        def new_repr(self) -> str:
            return verifier_key

        underlying_estimator_mock._verifier.__repr__ = new_repr

        cacher = TwoViewEstimatorCacher(two_view_estimator_obj=underlying_estimator_mock)

        result = cacher.run_2view(
            keypoints_i1=self.keypoints_i1,
            keypoints_i2=self.keypoints_i2,
            putative_corr_idxs=self.corr_idxs,
            camera_intrinsics_i1=self.camera_intrinsics_i1,
            camera_intrinsics_i2=self.camera_intrinsics_i2,
            i2Ti1_prior=None,
            gt_camera_i1=None,
            gt_camera_i2=None,
            gt_scene_mesh=None,
        )
        # Assert the returned value.
        self.assertEqual(result, self.dummy_output)

        # Assert that underlying TwoViewEstimator was called, to generate result from scratch (cache miss).
        underlying_estimator_mock.run_2view.assert_called_once()

        # Assert that hash generation was called once.
        generate_hash_for_numpy_array_mock.assert_called()

        # Assert that read function was called once and write function was called once.
        cache_path = ROOT_PATH / "cache" / "two_view_estimator" / f"{verifier_key}_numpy_key.pbz2"
        read_mock.assert_called_once_with(cache_path)
        write_mock.assert_called_once()

    @patch("gtsfm.utils.cache.generate_hash_for_numpy_array", return_value="numpy_key")
    @patch("gtsfm.utils.io.read_from_bz2_file", return_value=_DUMMY_OUTPUT)
    @patch("gtsfm.utils.io.write_to_bz2_file")
    def test_cache_hit(
        self, write_mock: MagicMock, read_mock: MagicMock, generate_hash_for_numpy_array_mock: MagicMock
    ) -> None:
        """Test the scenario of cache miss for TwoViewEstimator.

        The cache hit is actually ensure by the mocked return value of the `gtsfm.utils.io.read_from_bz2_file()`.
        The I/O function returns `None` on cache miss, and we mock it to return `None`.
        """

        # Mock the underlying two-view estimator which is used on cache miss.
        underlying_estimator_mock = MagicMock()
        underlying_estimator_mock.run_2view.return_value = self.dummy_output
        verifier_key = "Ransac__use_intrinsicsTrue_4px"

        def new_repr(self) -> str:
            return verifier_key

        underlying_estimator_mock._verifier.__repr__ = new_repr
        cacher = TwoViewEstimatorCacher(two_view_estimator_obj=underlying_estimator_mock)

        result = cacher.run_2view(
            keypoints_i1=self.keypoints_i1,
            keypoints_i2=self.keypoints_i2,
            putative_corr_idxs=self.corr_idxs,
            camera_intrinsics_i1=self.camera_intrinsics_i1,
            camera_intrinsics_i2=self.camera_intrinsics_i2,
            i2Ti1_prior=None,
            gt_camera_i1=None,
            gt_camera_i2=None,
            gt_scene_mesh=None,
        )

        # Assert the returned value.
        self.assertEqual(result, self.dummy_output)

        # Assert that underlying object was not called.
        underlying_estimator_mock.run_2view.assert_not_called()

        # Assert that hash generation was called with the inputs.
        generate_hash_for_numpy_array_mock.assert_called()

        # Assert that the read function was called once.
        cache_path = ROOT_PATH / "cache" / "two_view_estimator" / f"{verifier_key}_numpy_key.pbz2"
        read_mock.assert_called_once_with(cache_path)

        # Assert that the write function was not called (as cache is mocked to already exist).
        write_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
