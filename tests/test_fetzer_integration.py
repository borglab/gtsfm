"""Regression tests for the GTSAM focal solver and its pipeline integration."""

from unittest.mock import patch

import gtsam
import numpy as np
import pytest
from dask import compute, delayed

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer, BundleAdjustmentOptions
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.frontend.verifier.poselib_verifier import PoseLibVerifier
from gtsfm.multi_view_optimizer import _calibrate_view_graph_from_lists, rescore_inliers_fundamental
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


def test_delayed_list_adapter_forwards_geometry() -> None:
    """The multiview adapter must forward native geometry and return a list through Dask."""
    x1, x2, F = _synthetic_pair()
    intrinsics = [gtsam.Cal3Bundler(700, 0, 0, 400, 300) for _ in range(2)]
    corr = {(0, 1): np.column_stack([np.arange(len(x1))] * 2)}
    task = delayed(_calibrate_view_graph_from_lists)(
        corr,
        [Keypoints(x1), Keypoints(x2)],
        intrinsics,
        i2Fi1_dict={(0, 1): F},
    )
    refined, removed = compute(task, scheduler="single-threaded")[0]
    assert isinstance(refined, list)
    np.testing.assert_allclose([cal.fx() for cal in refined], 600, atol=1)
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


@pytest.mark.parametrize("scale", [1.0, -1.0, 1e-9])
@pytest.mark.parametrize("translation", [(1, 0, 0), (0, 1, 0)])
def test_rescoring_rectified_stereo(translation: tuple, scale: float) -> None:
    """Axis-aligned epipoles and arbitrary F scale must retain inliers and remove outliers."""
    rng = np.random.default_rng(0)
    points = rng.uniform([-1, -1, 4], [1, 1, 10], (40, 3))
    cal = gtsam.Cal3Bundler(600, 0, 0, 400, 300)
    camera = gtsam.PinholeCameraCal3Bundler(gtsam.Pose3(), cal)
    x1 = np.array([camera.project(point) for point in points])
    x2 = np.array([camera.project(point + translation) for point in points])
    x2[-1] += [100, 100]
    corr = np.column_stack([np.arange(len(x1))] * 2)
    rotation, direction = gtsam.Rot3(), gtsam.Unit3(np.array(translation, dtype=float))
    F = np.linalg.inv(cal.K()).T @ gtsam.EssentialMatrix(rotation, direction).matrix() @ np.linalg.inv(cal.K())
    args = ({(0, 1): rotation}, {(0, 1): direction}, {(0, 1): corr}, [Keypoints(x1), Keypoints(x2)], [cal, cal])
    rescored, rotations, directions = rescore_inliers_fundamental(
        *args,
        config_dict={(0, 1): 3},
        i2Fi1_dict={(0, 1): scale * F},
        min_inlier_count=30,
    )
    np.testing.assert_array_equal(rescored[(0, 1)], corr[:-1])
    assert (0, 1) in rotations and (0, 1) in directions
    rescored, rotations, directions = rescore_inliers_fundamental(
        *args,
        config_dict={(0, 1): 3},
        i2Fi1_dict={(0, 1): scale * F},
        min_inlier_count=40,
    )
    assert not rescored and not rotations and not directions


def test_failed_fundamental_is_not_retried_for_calibration() -> None:
    """A failed frontend F estimate must explicitly mark the pair ineligible for calibration."""
    verifier = PoseLibVerifier(estimate_calibration_geometry=True)
    with patch("gtsfm.frontend.verifier.poselib_verifier.poselib.estimate_fundamental", side_effect=RuntimeError):
        F, config = verifier._estimate_fundamental_and_config(np.zeros((40, 2)), np.zeros((40, 2)))
    assert F is None
    assert config not in vgc.FETZER_ALLOWED_CONFIGS and config is not None


@pytest.mark.parametrize("trim", [False, True])
def test_ba_measurement_filter_and_mask(trim: bool) -> None:
    """Opt-in trimming preserves long tracks and composes the original-track survival mask."""
    data = GtsfmData(3)
    cal = gtsam.Cal3Bundler(600, 0, 0, 400, 300)
    for i in range(3):
        data.add_camera(i, gtsam.PinholeCameraCal3Bundler(gtsam.Pose3(gtsam.Rot3(), np.array([i, 0, 0.0])), cal))
    for j in range(2):
        point = np.array([0, 0, 5.0])
        track = gtsam.SfmTrack(point)
        for i in range(3):
            uv = data.get_camera(i).project(point)
            if i >= 2 - j:
                uv = uv + 100
            track.addMeasurement(i, uv)
        data.add_track(track)
    ba = BundleAdjustmentOptions(trim_landmark_measurements=trim, min_tracks_per_camera=0).to_optimizer()
    with (
        patch.object(BundleAdjustmentOptimizer, "_BundleAdjustmentOptimizer__construct_factor_graph"),
        patch.object(
            BundleAdjustmentOptimizer,
            "_BundleAdjustmentOptimizer__optimize_and_recover",
            return_value=(data, None, 0.0, [True, True]),
        ),
    ):
        _, filtered, mask, _ = ba.run_ba_stage_with_filtering(
            data,
            [],
            {},
            reproj_error_thresh=5,
            verbose=False,
        )
    assert mask == [trim, False]
    assert filtered.number_tracks() == int(trim)
    if trim:
        assert filtered.get_track(0).numberMeasurements() == 2


@pytest.mark.parametrize("planar", [False, True])
def test_poselib_returns_calibration_geometry(planar: bool) -> None:
    """The real frontend returns pose, F and the expected planar classification."""
    x1, x2, _ = _synthetic_pair(planar=planar)
    corr = np.column_stack([np.arange(len(x1))] * 2)
    cal = gtsam.Cal3Bundler(600, 0, 0, 400, 300)
    verifier = PoseLibVerifier(estimate_calibration_geometry=True)
    rotation, direction, inliers, ratio, F, config = verifier.verify(Keypoints(x1), Keypoints(x2), corr, cal, cal)
    assert rotation is not None and direction is not None
    assert len(inliers) >= 30 and ratio > 0.9
    assert F.shape == (3, 3)
    assert config == (6 if planar else 3)
    failure = verifier.verify(Keypoints(x1), Keypoints(x2), corr[:2], cal, cal)
    assert len(failure) == 6 and failure[0] is None and failure[4] is None


def test_phototourism_config_and_two_view_geometry_plumbing() -> None:
    """The new configuration must instantiate and carry verifier geometry to multiview calibration."""
    from pathlib import Path

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from gtsfm.multi_view_optimizer import MultiViewOptimizer

    config_path = Path(__file__).parents[1] / "gtsfm/configs/megaloc_sift_gp_single_pt.yaml"
    config = OmegaConf.load(config_path)
    optimizer = instantiate(config.cluster_optimizer.multiview_optimizer)
    assert optimizer._rescore_inliers and not optimizer._run_second_view_graph_estimator
    assert optimizer.ba_optimizer._trim_landmark_measurements
    # Skip BA here to isolate metadata transport from the independently tested BA stage.
    estimator = instantiate(
        config.cluster_optimizer.two_view_estimator.two_view_estimator_obj, bundle_adjust_2view=False
    )
    x1, x2, _ = _synthetic_pair()
    corr = np.column_stack([np.arange(len(x1))] * 2)
    cal = gtsam.Cal3Bundler(600, 0, 0, 400, 300)
    result = estimator.run_2view(Keypoints(x1), Keypoints(x2), corr, cal, cal, None, None, None, i1=0, i2=1)
    components = MultiViewOptimizer._extract_two_view_components({(0, 1): result})
    np.testing.assert_array_equal(components[5][(0, 1)], result.i2Fi1)
    assert components[6][(0, 1)] == 3
