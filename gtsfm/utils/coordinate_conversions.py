"""Utility functions for geometric coordinate conversions.

Authors: Akshay Krishnan
"""
from typing import List

import numpy as np
from gtsam import Unit3


def cartesian_to_spherical_directions(directions: List[Unit3]) -> np.ndarray:
    """Converts a list of Unit3 directions to spherical coordinates (azimuth, elevation) in radians.
    Angles are given in a compass frame:
    zenith is along +y, azimuth=0 is along +z, and azimuth=pi/2 is along +x.
    This follows 1DSfM's convention
    https://github.com/wilsonkl/SfM_Init/blob/f801a4ace3b34f990cbda3c57b96387ce19c90c1/sfminit/onedsfm.py#L132

    Args:
        directions: List of Unit3 directions.

    Returns:
        Nx2 numpy array where N is the length of the list and columns are (azimuth, elevation) in radians.
    """
    directions_array = np.array([d.point3() for d in directions])
    azimuth = np.arctan2(directions_array[:, 0], directions_array[:, 2])
    elevation = np.arccos(directions_array[:, 1])
    return np.column_stack((azimuth, elevation))


def spherical_to_cartesian_directions(spherical_coords: np.ndarray) -> List[Unit3]:
    """Converts an array of spherical coordinates to a list of Unit3 directions.

    Args:
        directions: Nx2 array where the first column are [azimuth, elevation] in radians.

    Returns:
        List of Unit3 directions for the provided spherical coordinates.
    """
    azimuth = spherical_coords[:, 0]
    elevation = spherical_coords[:, 1]
    y = np.cos(elevation)
    x = np.sin(azimuth) * np.sin(elevation)
    z = np.cos(azimuth) * np.sin(elevation)
    directions_unit3 = []

    for i in range(x.shape[0]):
        directions_unit3.append(Unit3(np.array([x[i], y[i], z[i]])))
    return directions_unit3
