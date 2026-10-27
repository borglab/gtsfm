"""Test the experiment-only bounded rotation solver."""

import gtsam
import numpy as np

from scripts.benchmark_rotation_averaging import UncertifiedRotationAveraging


def test_uncertified_solver_recovers_relative_rotations() -> None:
    """Recover a connected set of exact relative rotations up to a global gauge."""
    rotations = [gtsam.Rot3(), gtsam.Rot3.RzRyRx(0.1, -0.2, 0.3), gtsam.Rot3.RzRyRx(-0.2, 0.1, 0.4)]
    pairs = [(0, 1), (0, 2), (1, 2)]
    measurements = gtsam.BinaryMeasurementsRot3()
    noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    for i, j in pairs:
        measurements.append(gtsam.BinaryMeasurementRot3(j, i, rotations[j].between(rotations[i]), noise))

    estimated = UncertifiedRotationAveraging()._run_with_consecutive_ordering(len(rotations), measurements)
    assert all(rotation is not None for rotation in estimated)
    for i, j in pairs:
        expected = rotations[j].between(rotations[i])
        actual = estimated[j].between(estimated[i])
        assert np.linalg.norm(gtsam.Rot3.Logmap(expected.between(actual))) < 1e-6
