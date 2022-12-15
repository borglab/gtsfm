"""Utilities for sampling/generating data on planar surfaces.

Authors: Ayush Baid, John Lambert, Akshay Krishnan
"""
from typing import List, Tuple

import numpy as np
from gtsam import Unit3
from scipy import stats

import gtsfm.utils.coordinate_conversions as conversion_utils


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


def sample_random_directions(num_samples: int) -> List[Unit3]:
    """Samples `num_samples` Unit3 3D directions, uniformly in Cartesian coordinates.

    Args:
        num_samples: Number of samples required.

    Returns:
        List of sampled Unit3 directions.
    """
    samples = np.random.normal(size=(num_samples, 3))
    return [Unit3(sample) for sample in samples]


def sample_kde_directions(measurements: List[Unit3], num_samples: int, max_kde_samples=2000) -> List[Unit3]:
    """Fits a Gaussian density kernel to the provided measurements, and then samples num_samples from this kernel.

    Args:
        w_i2Ui1_measurements: List of BinaryMeasurementUnit3 direction measurements.
        num_samples: Number of samples to be sampled from the kernel.

    Returns:
        List of sampled Unit3 directions.
    """
    if len(measurements) > max_kde_samples:
        sampled_idx = np.random.choice(len(measurements), max_kde_samples, replace=False).tolist()
        measurements_subset = [measurements[i] for i in sampled_idx]
    else:
        measurements_subset = measurements

    measurements_spherical = conversion_utils.cartesian_to_spherical_directions(measurements_subset)

    # gaussian_kde expects each sample to be a column, hence transpose.
    kde = stats.gaussian_kde(measurements_spherical.T)
    sampled_directions_spherical = kde.resample(size=num_samples).T
    return conversion_utils.spherical_to_cartesian_directions(sampled_directions_spherical)
