"""Utilities to support triangulation.

Authors: Ayush Baid
"""
import numpy as np
from gtsam import PinholeCameraCal3Bundler, Unit3

import gtsfm.utils.geometry_comparisons as geometry_utils


def calculate_triangulation_angle_in_degrees(
    camera_1: PinholeCameraCal3Bundler, camera_2: PinholeCameraCal3Bundler, point_3d: np.ndarray
) -> float:
    """Calculates the angle formed at the 3D point by the rays backprojected from 2 cameras.

    In the setup with X (point_3d) and two cameras C1 and C2, the triangulation angle is the angle between rays C1->X
    and C2->X, i.e. the angle subtended at the 3d point.
        X
       / \
      /   \
     /     \
    C1      C2

    References:
    - https://github.com/colmap/colmap/blob/dev/src/base/triangulation.cc#L122

    Args:
        camera_1: the first camera.
        camera_2: the second camera.
        point_3d: a 3d point which is imaged by the two cameras.

    Returns:
        the angle formed between the two rays, at the 3d point, in degrees.
    """
    camera_center_1: np.ndarray = camera_1.pose().translation()
    camera_center_2: np.ndarray = camera_2.pose().translation()

    # compute the two rays
    ray_1 = point_3d - camera_center_1
    ray_2 = point_3d - camera_center_2

    return geometry_utils.compute_relative_unit_translation_angle(Unit3(ray_1), Unit3(ray_2))