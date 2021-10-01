"""Unit tests for the transformation utils.

Authors: Akshay Krishnan
"""
from gtsam import Unit3
import numpy as np

import gtsfm.utils.transforms as transform_utils

def test_convert_euclidean_to_spherical_directions() -> None:
    directions = [
        Unit3(np.array([1, 0, 0])),
        Unit3(np.array([0, 1, 0])),
        Unit3(np.array([0, 0, 1])),
    ]
    expected_spherical_coordinates = np.deg2rad(np.array([
        [90, 90], 
        [180, 0],
        [180, 90]
        ]))
    spherical_coordinates = transform_utils.euclidean_to_spherical_directions(directions)
    np.testing.assert_allclose(spherical_coordinates, expected_spherical_coordinates)


def test_convert_spherical_to_euclidean_directions() -> None:
    directions = [
        Unit3(np.array([1, 0, 0])),
        Unit3(np.array([0, 1, 0])),
        Unit3(np.array([0, 0, 1])),
        Unit3(np.array([1, 0, 1])),
        Unit3(np.array([0, 1, 1])),
    ]
    spherical_coordinates = transform_utils.euclidean_to_spherical_directions(directions)
    euclidean_directions = transform_utils.spherical_to_euclidean_directions(spherical_coordinates)
    for i in range(len(euclidean_directions)):
        np.testing.assert_almost_equal(euclidean_directions[i].point3(), directions[i].point3())
