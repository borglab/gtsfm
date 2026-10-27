"""Check the benchmark's fixed-denominator pose evaluation."""

import numpy as np
import pytest
from gtsam import Pose3, Rot3

from scripts.evaluate_sift_comparison import evaluate


def test_missing_camera_counts_as_failed_pairs() -> None:
    """Two perfect poses out of three contribute only one of three possible pairs."""
    reference = {str(i): Pose3(Rot3(), np.array([float(i), 0, 0])) for i in range(3)}
    result = evaluate(reference, {name: pose for name, pose in reference.items() if name != "2"})
    assert result["evaluated_pairs"] == 3
    assert result["missing_pairs"] == 2
    assert result["registered_images"] == 2
    assert result["missing_images"] == ["2"]
    for value in result["auc_percent"].values():
        assert value == pytest.approx(100 / 3, abs=1e-4)


def test_relative_pose_scores_are_similarity_invariant() -> None:
    """Global rotation, translation, scale, and dictionary order do not alter accuracy."""
    reference = {str(i): Pose3(Rot3.Rz(i * 0.1), np.array([float(i), i * i, 0])) for i in range(4)}
    rotation = Rot3.RzRyRx(0.1, 0.2, 0.3)
    estimated = {
        name: Pose3(rotation.compose(pose.rotation()), 3 * rotation.rotate(pose.translation()) + np.array([4, 5, 6]))
        for name, pose in reversed(list(reference.items()))
    }
    result = evaluate(reference, estimated)
    assert result["evaluated_pairs"] == 6
    assert result["missing_pairs"] == 0
    for value in result["auc_percent"].values():
        assert value == pytest.approx(100, abs=1e-3)
