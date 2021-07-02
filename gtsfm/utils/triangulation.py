"""Utilities to support triangulation.

Authors: Ayush Baid
"""
import numpy as np
from gtsam import PinholeCameraCal3Bundler


def calculate_triangulation_angle_in_degrees(
    camera_1: PinholeCameraCal3Bundler, camera_2: PinholeCameraCal3Bundler, point_3d: np.ndarray
) -> float:
    """Calculates the angle formed at the 3D point by the rays backprojected from 2 cameras.

    Args:
        camera_1: the first camera.
        camera_2: the second camera.
        point_3d: the 3d point to compute the angle at.

    Returns:
        the angle formed at the 3d point, in radians.
    """
    camera_center_1: np.ndarray = camera_1.pose().translation()
    camera_center_2: np.ndarray = camera_2.pose().translation()

    # compute the three squared edge lengths of the triangle formed between camera centers and the 3d point
    def squared_dist_fn(x, y):
        return np.sum(np.square(x - y), axis=None)

    baseline_squared = squared_dist_fn(camera_center_1, camera_center_2)
    ray_length_1_squared = squared_dist_fn(camera_center_1, point_3d)
    ray_length_2_squared = squared_dist_fn(camera_center_2, point_3d)

    # use the law of cosines to estimate the angle at the 3d point
    numerator = ray_length_1_squared + ray_length_2_squared - baseline_squared
    denominator = 2 * np.sqrt(ray_length_1_squared * ray_length_2_squared)
    if denominator == 0:
        return 0
    angle_radians = np.arccos(numerator / denominator)

    return np.rad2deg(angle_radians)

