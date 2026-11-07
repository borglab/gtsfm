"""Sparse gauge regularization for monocular bundle adjustment."""

from collections.abc import Sequence

import gtsam
import numpy as np
from scipy.spatial.distance import cdist


def add_gauge_factors(
    graph: gtsam.NonlinearFactorGraph,
    initial_values: gtsam.Values,
    point_keys: Sequence[int],
    camera_keys: Sequence[int],
    *,
    point_sigma: float = 100.0,
    camera_y_sigma: float = 100.0,
    camera_z_sigma: float = 10.0,
    baseline_sigma: float = 0.01,
) -> None:
    """Add soft priors on landmark norms, camera-center Y/Z, and the longest initial camera baseline.

    The point and camera priors jointly choose the origin. About that origin, the camera
    penalties favor the longest trajectory axis along X and the shortest along Z, up to sign.
    The soft priors can perturb shape and allow some baseline shrinkage.

    Args:
        graph: Bundle-adjustment graph to extend in place.
        initial_values: Initial camera-to-world Pose3 values used to measure the baseline.
        point_keys: Keys of the Point3 landmarks.
        camera_keys: Keys of the camera-to-world poses.
        point_sigma: Sigma for RMS landmark distance from the origin, in units of the initial baseline.
        camera_y_sigma: Sigma for the RMS camera-center Y coordinate, in the same units.
        camera_z_sigma: Sigma for the RMS camera-center Z coordinate. Must be smaller than camera_y_sigma.
        baseline_sigma: Sigma for the fractional change in baseline length.
    """
    if len(point_keys) == 0 or len(camera_keys) < 2:
        raise ValueError("Gauge factors require at least one landmark and two cameras.")
    if not all(0 < sigma < np.inf for sigma in (point_sigma, camera_y_sigma, camera_z_sigma, baseline_sigma)):
        raise ValueError("Gauge sigmas must be finite and positive.")
    if camera_z_sigma >= camera_y_sigma:
        raise ValueError("camera_z_sigma must be smaller than camera_y_sigma.")

    centers = np.array([initial_values.atPose3(key).translation() for key in camera_keys])
    if not np.isfinite(centers).all():
        raise ValueError("Camera centers must be finite.")

    # Scan one row of pairwise distances at a time to keep temporary storage linear in the number of cameras.
    baseline = 0.0
    for i in range(len(centers) - 1):
        distances = cdist(centers[i : i + 1], centers[i + 1 :])[0]
        offset = int(np.argmax(distances))
        if distances[offset] > baseline:
            baseline = distances[offset]
            baseline_keys = (camera_keys[i], camera_keys[i + 1 + offset])
    if baseline == 0:
        raise ValueError("At least two camera centers must be distinct to define scale.")

    # Average each unary cost over its key set to keep its weight independent of scene size.
    point_noise = gtsam.noiseModel.Isotropic.Sigma(3, point_sigma * baseline * np.sqrt(len(point_keys)))
    precisions = np.array([0.0, camera_y_sigma**-2, camera_z_sigma**-2]) / (len(camera_keys) * baseline**2)
    camera_noise = gtsam.noiseModel.Diagonal.Precisions(precisions)
    baseline_noise = gtsam.noiseModel.Isotropic.Sigma(1, baseline_sigma * baseline)

    origin = np.zeros(3)
    for key in point_keys:
        graph.add(gtsam.PriorFactorPoint3(key, origin, point_noise))
    for key in camera_keys:
        graph.add(gtsam.GPSFactor(key, origin, camera_noise))
    graph.add(gtsam.RangeFactorPose3(*baseline_keys, baseline, baseline_noise))
