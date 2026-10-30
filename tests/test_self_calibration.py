"""Regression tests for GTSAM view-graph self-calibration."""

from unittest.mock import patch

import gtsam
import numpy as np
import pytest

from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.view_graph_estimator import view_graph_calibration as vgc
from tests.test_view_graph_calibration_robust import _synthetic_pair


def test_sparse_camera_ids_and_plumbed_fundamental() -> None:
    """The native factor must refine sparse camera IDs without re-estimating a supplied F."""
    x1, x2, F = _synthetic_pair()
    intrinsics = {i: gtsam.Cal3Bundler(700, 0, 0, 400, 300) for i in (3, 12, 25)}
    keypoints = {3: Keypoints(x1), 12: Keypoints(x2)}
    corr = {(3, 12): np.column_stack([np.arange(len(x1))] * 2)}
    with patch.object(vgc, "_estimate_fundamental_robust", side_effect=AssertionError("F already supplied")):
        refined, removed = vgc.calibrate_view_graph(corr, keypoints, intrinsics, i2Fi1_dict={(3, 12): F})
    assert set(refined) == set(intrinsics)
    assert refined[25] is intrinsics[25]
    np.testing.assert_allclose([refined[i].fx() for i in (3, 12)], 600, atol=1)
    assert not removed


def test_empty_calibration_preserves_dictionary() -> None:
    """No eligible edges must preserve calibration objects and sparse keys."""
    intrinsics = {12: gtsam.Cal3Bundler(700, 0, 0, 400, 300)}
    refined, removed = vgc.calibrate_view_graph({}, {}, intrinsics)
    assert refined == intrinsics
    assert not removed


@pytest.mark.parametrize("config", [1, 4, 5, 6])
def test_ineligible_config_preserves_pose_edges(config: int) -> None:
    """A rejected calibration configuration must not remove its useful pose edge."""
    x1, x2, F = _synthetic_pair()
    intrinsics = {i: gtsam.Cal3Bundler(700, 0, 0, 400, 300) for i in (0, 1)}
    corr = {(0, 1): np.column_stack([np.arange(len(x1))] * 2)}
    refined, removed = vgc.calibrate_view_graph(
        corr,
        {0: Keypoints(x1), 1: Keypoints(x2)},
        intrinsics,
        i2Fi1_dict={(0, 1): F},
        config_dict={(0, 1): config},
    )
    assert refined == intrinsics
    assert not removed


@pytest.mark.parametrize(
    "calibration",
    [
        gtsam.Cal3Bundler(700, 0.1, 0.02, 400, 300),
        gtsam.Cal3_S2(700, 700, 0.1, 400, 300),
        gtsam.Cal3DS2(700, 700, 0.1, 400, 300, 0.1, 0.02, 0.03, 0.04),
        gtsam.Cal3Fisheye(700, 700, 0.1, 400, 300, 0.1, 0.02, 0.03, 0.04),
    ],
)
def test_refit_preserves_calibration_model(calibration: CALIBRATION_TYPE) -> None:
    """Focal replacement must preserve distortion, skew and principal point in every supported model."""
    refined = vgc._refit_focal(calibration, 600)
    assert type(refined) is type(calibration)
    assert refined.fx() == refined.fy() == 600
    for name in ("px", "py", "skew", "k1", "k2", "k3", "k4", "p1", "p2"):
        if hasattr(calibration, name):
            assert getattr(refined, name)() == getattr(calibration, name)()
