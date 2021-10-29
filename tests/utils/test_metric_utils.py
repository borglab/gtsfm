"""Unit tests for metrics utilities.

Author: Travis Driver
"""

import unittest

import trimesh
import numpy as np
from gtsam import PinholeCameraCal3Bundler, Cal3Bundler
from gtsfm.common.keypoints import Keypoints

import gtsfm.utils.metrics as metric_utils


class TestMetricUtils(unittest.TestCase):
    """Class containing all unit tests for metric utils."""

    def test_mesh_inlier_correspondences(self):
        """Tests `compute_keypoint_intersections()` function."""
        # Create cube mesh with side length one centered at origin.
        box = trimesh.primitives.Box()

        # Create arrangement of two camera pointing at center of one of the cube's faces.
        calibration = Cal3Bundler(10, 0, 0, 1, 1)
        target_pos = [0.5, 0, 0]
        cam_i1 = PinholeCameraCal3Bundler().Lookat([2, 1, 0], target_pos, [0, -1, 0], calibration)
        cam_i2 = PinholeCameraCal3Bundler().Lookat([2, -1, 0], target_pos, [0, -1, 0], calibration)
        keypoints_i1 = Keypoints(coordinates=np.array([[1, 1]]).astype(np.float32))
        keypoints_i2 = Keypoints(coordinates=np.array([[1, 1]]).astype(np.float32))

        # Project keypoint at center of each simulated image and record intersection.
        is_inlier, reproj_err = metric_utils.mesh_inlier_correspondences(
            keypoints_i1, keypoints_i2, cam_i1, cam_i2, box, 0.1
        )
        assert np.count_nonzero(is_inlier) == 1
        assert reproj_err[0] < 1e-4

    def test_compute_keypoint_intersections(self):
        """Tests `compute_keypoint_intersections()` function."""
        # Create cube mesh with side length one centered at origin.
        box = trimesh.primitives.Box()

        # Create arrangement of 4 cameras in x-y plane pointing at the cube.
        calibration = Cal3Bundler(10, 0, 0, 1, 1)
        cam_pos = [[2, 0, 0], [-2, 0, 0], [0, 0, 2], [0, 0, -2]]
        target_pos = [0, 0, 0]
        cams = [PinholeCameraCal3Bundler().Lookat(c, target_pos, [0, -1, 0], calibration) for c in cam_pos]

        # Project keypoint at center of each simulated image and record intersection.
        kpt = Keypoints(coordinates=np.array([[1, 1]]).astype(np.float32))
        expected_intersections = [[0.5, 0, 0], [-0.5, 0, 0], [0, 0, 0.5], [0, 0, -0.5]]
        estimated_intersections = []
        for cam in cams:
            _, intersection = metric_utils.compute_keypoint_intersections(kpt, cam, box, verbose=True)
            estimated_intersections.append(intersection.flatten().tolist())
        np.testing.assert_allclose(expected_intersections, estimated_intersections)


if __name__ == "__main__":
    unittest.main()
