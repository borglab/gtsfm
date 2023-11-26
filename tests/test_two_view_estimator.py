"""Unit tests for the two-view estimator.

Authors: Ayush Baid
"""
import unittest

import gtsam
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, PinholeCameraCal3Bundler, Pose3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.io as io_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.two_view_estimator import TwoViewEstimator
from gtsfm.data_association.point3d_initializer import TriangulationOptions, TriangulationSamplingMode

GTSAM_EXAMPLE_FILE = "5pointExample1.txt"
EXAMPLE_DATA = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestTwoViewEstimator(unittest.TestCase):
    """Unit tests for the 2-view estimator.

    Uses GTSAM's 5-point example for ground truth tracks and cameras. See `gtsam/examples/Data/5pointExample1.txt` for
    details.
    """

    def setUp(self):
        """Create keypoints."""
        num_points = 5
        normalized_coordinates_i1 = []
        normalized_coordinates_i2 = []
        for i in range(num_points):
            track = EXAMPLE_DATA.get_track(i)
            normalized_coordinates_i1.append(track.measurement(0)[1])
            normalized_coordinates_i2.append(track.measurement(1)[1])
        normalized_coordinates_i1 = np.array(normalized_coordinates_i1)
        normalized_coordinates_i2 = np.array(normalized_coordinates_i2)
        self.keypoints_i1 = Keypoints(normalized_coordinates_i1)
        self.keypoints_i2 = Keypoints(normalized_coordinates_i2)
        self.corr_idxs = np.hstack([np.arange(5).reshape(-1, 1)] * 2)

        self.two_view_estimator = TwoViewEstimator(
            verifier=None,
            triangulation_options=TriangulationOptions(mode=TriangulationSamplingMode.NO_RANSAC),
            inlier_support_processor=None,
            bundle_adjust_2view=True,
            eval_threshold_px=4,
            allow_indeterminate_linear_system=True,
        )

    def test_two_view_correspondences(self):
        """Tests the bundle adjustment for relative pose on a simulated scene."""
        i1Ri2 = EXAMPLE_DATA.get_camera(1).pose().rotation()
        i1ti2 = EXAMPLE_DATA.get_camera(1).pose().translation()
        i2Ti1 = Pose3(i1Ri2, i1ti2)
        cameras = {
            0: PinholeCameraCal3Bundler(Pose3(), Cal3Bundler()),
            1: PinholeCameraCal3Bundler(i2Ti1, Cal3Bundler()),
        }

        tracks_3d, valid_indices = self.two_view_estimator.triangulate_two_view_correspondences(
            cameras, self.keypoints_i1, self.keypoints_i2, self.corr_idxs
        )
        self.assertEqual(len(tracks_3d), 5)
        self.assertEqual(len(valid_indices), 5)

    def test_bundle_adjust(self):
        """Tests the bundle adjustment for relative pose on a simulated scene."""
        # Extract example poses.
        i1Ri2 = EXAMPLE_DATA.get_camera(1).pose().rotation()
        i1ti2 = EXAMPLE_DATA.get_camera(1).pose().translation()
        i2Ti1 = Pose3(i1Ri2, i1ti2).inverse()
        i2Ei1 = EssentialMatrix(i2Ti1.rotation(), Unit3(i2Ti1.translation()))

        # Perform bundle adjustment.
        i2Ri1_optimized, i2Ui1_optimized, corr_idxs = self.two_view_estimator.bundle_adjust(
            keypoints_i1=self.keypoints_i1,
            keypoints_i2=self.keypoints_i2,
            verified_corr_idxs=self.corr_idxs,
            camera_intrinsics_i1=Cal3Bundler(),
            camera_intrinsics_i2=Cal3Bundler(),
            i2Ri1_initial=i2Ei1.rotation(),
            i2Ui1_initial=i2Ei1.direction(),
            i2Ti1_prior=None,
        )

        # Assert
        rotation_angular_error = comp_utils.compute_relative_rotation_angle(i2Ri1_optimized, i2Ei1.rotation())
        translation_angular_error = comp_utils.compute_relative_unit_translation_angle(
            i2Ui1_optimized, i2Ei1.direction()
        )
        self.assertLessEqual(rotation_angular_error, 1)
        self.assertLessEqual(translation_angular_error, 1)
        np.testing.assert_allclose(corr_idxs, self.corr_idxs)


if __name__ == "__main__":
    unittest.main()
