"""Utilities for sampling/generating data on planar surfaces.

Authors: Ayush Baid, John Lambert
"""
from typing import Tuple

import numpy as np


def sample_points_on_plane(
    plane_coefficients: Tuple[float, float, float, float],
    range_x: Tuple[float, float],
    range_y: Tuple[float, float],
    num_points: int,
) -> np.ndarray:
    """Sample random points on a 3D plane ax + by + cz + d = 0.

    Args:
        plane_coefficients: coefficients (a,b,c,d) of the plane equation.
        range_x: desired range of the x coordinates of samples.
        range_y: desired range of the y coordinates of samples.
        num_points: number of points to sample.

    Returns:
        3d points on the plane, of shape (num_points, 3).
    """

    a, b, c, d = plane_coefficients

    if c == 0:
        raise ValueError("z-coefficient for the plane should not be zero")

    # sample x and y coordinates randomly
    x = np.random.uniform(low=range_x[0], high=range_x[1], size=(num_points, 1))
    y = np.random.uniform(low=range_y[0], high=range_y[1], size=(num_points, 1))

    # calculate z coordinates using equation of the plane
    z = -(a * x + b * y + d) / c

    pts = np.hstack([x, y, z])
    return pts
