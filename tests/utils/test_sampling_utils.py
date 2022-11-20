"""Unit test on utility for sampling/generating data on planar surfaces.

Authors: Ayush Baid, John Lambert, Akshay Krishnan
"""

import numpy as np

from gtsam import Unit3

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


def test_sample_random_directions() -> None:
    num_samples = 1000

    direction_list = sampling_utils.sample_random_directions(num_samples)
    assert len(direction_list) == num_samples

    directions = np.array([direction.point3() for direction in direction_list])
    mean_direction = np.mean(directions, axis=0)
    std_direction = np.std(directions, axis=0)
    std_direction_relative = std_direction / std_direction[0]

    # Increase number of samples for a smaller threshold
    assert np.allclose(mean_direction, np.zeros(3), atol=1e-1)
    assert np.allclose(std_direction_relative, np.ones(3), atol=1e-1)


def test_sample_kde_directions() -> None:
    num_samples = 1000

    # centered at [1, 1, 1] with variance 0.1
    mean = np.array([1, 1, 1])
    cov = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    samples = np.random.multivariate_normal(mean, cov, num_samples)
    samples_normd = [Unit3(sample) for sample in samples]

    actual_samples = sampling_utils.sample_kde_directions(samples_normd, num_samples)
    assert len(actual_samples) == num_samples
    
    actual_samples_array = np.array([sample.point3() for sample in actual_samples])
    samples_mean = np.mean(actual_samples_array, axis=0)
    samples_mean_relative = samples_mean / samples_mean[0]
    samples_cov = np.cov(actual_samples_array.T)

    assert np.allclose(samples_mean_relative, mean, atol=1e-1)
    assert np.allclose(samples_cov, cov, atol=1e-1)
