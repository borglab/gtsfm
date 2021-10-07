"""Unit test on utility for sampling/generating data on planar surfaces.

Authors: Ayush Baid, John Lambert
"""

import numpy as np

import gtsfm.utils.sampling as sampling_utils


def test_sample_points_on_plane() -> None:
    """Assert generated points are on a single 3d plane."""

    num_points = 10

    # range of x and y coordinates for 3D points
    range_x = (-7, 7)
    range_y = (-10, 10)

    # define the plane equation
    # plane at z=10, so ax + by + cz + d = 0 + 0 + -z + 10 = 0
    plane_coefficients = (0, 0, -1, 10)

    pts = sampling_utils.sample_points_on_plane(plane_coefficients, range_x, range_y, num_points)

    # ensure ax + by + cz + d = 0
    pts_residuals = pts @ np.array(plane_coefficients[:3]).reshape(3, 1) + plane_coefficients[3]
    np.testing.assert_almost_equal(pts_residuals, np.zeros((num_points, 1)))

    assert pts.shape == (10, 3)
