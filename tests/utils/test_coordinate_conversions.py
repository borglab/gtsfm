"""Unit tests for the coordinate conversion utils.

Authors: Akshay Krishnan
"""
import numpy as np
from gtsam import Unit3

import gtsfm.utils.coordinate_conversions as conversion_utils


def test_convert_cartesian_to_spherical_directions() -> None:
    """Check that correct spherical coordinates are obtained for certain directions."""
    directions = [
        Unit3(np.array([1, 0, 0])),
        Unit3(np.array([0, 1, 0])),
        Unit3(np.array([0, 0, 1])),
    ]
    expected_spherical_coordinates = np.deg2rad(np.array([[90, 90], [0, 0], [0, 90]]))
    spherical_coordinates = conversion_utils.cartesian_to_spherical_directions(directions)
    np.testing.assert_allclose(spherical_coordinates, expected_spherical_coordinates)


def test_convert_spherical_to_cartesian_directions() -> None:
    """Check that Euclidean -> Spherical -> Euclidean conversion yields same results."""
    directions = [
        Unit3(np.array([1, 0, 0])),
        Unit3(np.array([0, 1, 0])),
        Unit3(np.array([0, 0, 1])),
        Unit3(np.array([1, 0, 1])),
        Unit3(np.array([0, 1, 1])),
    ]
    spherical_coordinates = conversion_utils.cartesian_to_spherical_directions(directions)
    cartesian_directions = conversion_utils.spherical_to_cartesian_directions(spherical_coordinates)
    for i in range(len(cartesian_directions)):
        np.testing.assert_almost_equal(cartesian_directions[i].point3(), directions[i].point3())
