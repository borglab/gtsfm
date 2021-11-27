"""Unit tests for metrics utilities.

Author: Travis Driver
"""

import unittest

import numpy as np
import trimesh
from gtsam import PinholeCameraCal3Bundler, Cal3Bundler

import gtsfm.utils.metrics as metric_utils
from gtsfm.common.keypoints import Keypoints


class TestMetricUtils(unittest.TestCase):
    """Class containing all unit tests for metric utils."""

    def test_mesh_inlier_correspondences(self) -> None:
        """Tests `compute_keypoint_intersections()` function.

        We arrange four cameras in the x-z plane around a cube centered at the origin with side length 1. These cameras
        are placed at (2, 0, 0), (-2, 0, 0), (0, 0, 2) and (0, 0, -2). We project a single 3d point located at the
        origin into each camera. Since the cube has unit length on each dimension, we expect a keypoint located at the
        center of each image to be found at the boundary of the cube -- 0.5 meters from the origin for each side on the
        z-x plane.
        """
        # Create cube mesh with side length one centered at origin.
        box = trimesh.primitives.Box()

        # Create arrangement of two cameras pointing at the center of one of the cube's faces.
        fx, k1, k2, u0, v0 = 10, 0, 0, 1, 1
        calibration = Cal3Bundler(fx, k1, k2, u0, v0)
        cam_pos = [[2, 1, 0], [2, -1, 0]]
        target_pos = [0.5, 0, 0]
        up_vector = [0, -1, 0]
        cam_i1 = PinholeCameraCal3Bundler().Lookat(cam_pos[0], target_pos, up_vector, calibration)
        cam_i2 = PinholeCameraCal3Bundler().Lookat(cam_pos[1], target_pos, up_vector, calibration)
        keypoints_i1 = Keypoints(coordinates=np.array([[1, 1]]).astype(np.float32))
        keypoints_i2 = Keypoints(coordinates=np.array([[1, 1]]).astype(np.float32))

        # Project keypoint at center of each simulated image and record intersection.
        is_inlier, reproj_err = metric_utils.mesh_inlier_correspondences(
            keypoints_i1, keypoints_i2, cam_i1, cam_i2, box, dist_threshold=0.1
        )
        assert np.count_nonzero(is_inlier) == 1
        assert reproj_err[0] < 1e-4

    def test_compute_keypoint_intersections(self) -> None:
        """Tests `compute_keypoint_intersections()` function."""
        # Create cube mesh with side length one centered at origin.
        box = trimesh.primitives.Box()

        # Create arrangement of 4 cameras in x-z plane pointing at the cube.
        fx, k1, k2, u0, v0 = 10, 0, 0, 1, 1
        calibration = Cal3Bundler(fx, k1, k2, u0, v0)
        cam_pos = [[2, 0, 0], [-2, 0, 0], [0, 0, 2], [0, 0, -2]]
        target_pos = [0, 0, 0]
        up_vector = [0, -1, 0]
        cams = [PinholeCameraCal3Bundler().Lookat(c, target_pos, up_vector, calibration) for c in cam_pos]

        # Project keypoint at center of each simulated image and record intersection.
        kpt = Keypoints(coordinates=np.array([[1, 1]]).astype(np.float32))
        expected_intersections = [[0.5, 0, 0], [-0.5, 0, 0], [0, 0, 0.5], [0, 0, -0.5]]
        estimated_intersections = []
        for cam in cams:
            _, intersection = metric_utils.compute_keypoint_intersections(kpt, cam, box, verbose=True)
            estimated_intersections.append(intersection.flatten().tolist())
        np.testing.assert_allclose(expected_intersections, estimated_intersections)


def test_compute_percentage_change_improve() -> None:
    """Ensure that percentage change is computed correctly for a 50% improvement over 100."""
    x = 100
    y = 150
    change_percent = metric_utils.compute_percentage_change(x, y)
    assert np.isclose(change_percent, 50)


def test_compute_percentage_change_static() -> None:
    """Ensure that percentage change is computed correctly for no change in a value."""
    x = 100
    y = 100
    change_percent = metric_utils.compute_percentage_change(x, y)
    assert np.isclose(change_percent, 0)


def test_compute_percentage_change_regression() -> None:
    """Ensure that percentage change is computed correctly for a 99% regression against 100."""
    x = 100
    y = 1
    change_percent = metric_utils.compute_percentage_change(x, y)
    assert np.isclose(change_percent, -99)


if __name__ == "__main__":
    unittest.main()
