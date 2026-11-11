"""Original-graph camera marginalization, gauge preservation, and sparse factorization."""

from pathlib import Path
from unittest.mock import patch

import gtsam
import numpy as np
import pytest
from gtsam.symbol_shorthand import K, L, X

from gtsfm import camera_tree_optimizer as cto
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils.tree import Tree
from tests.test_camera_tree_optimizer import _assert_same_solution, _overlapping_lund_tree, _synthetic_scene


def _add_graph_gauge(graph: gtsam.NonlinearFactorGraph, values: gtsam.Values, camera_keys: list[int]) -> None:
    """Add exact pose and scale constraints directly to an original nonlinear graph."""
    anchor_key = camera_keys[0]
    anchor = values.atPose3(anchor_key)
    farthest = max(camera_keys[1:], key=lambda key: anchor.range(values.atPose3(key)))
    graph.push_back(gtsam.NonlinearEqualityPose3(anchor_key, anchor))
    graph.push_back(
        gtsam.RangeFactorPose3(
            anchor_key, farthest, anchor.range(values.atPose3(farthest)), gtsam.noiseModel.Constrained.All(1)
        )
    )


def _original_leaf_graph(scene: GtsfmData) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
    """Build a conventional Pose3/point/calibration BA graph with its original gauge."""
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    calibration = scene.get_camera(0).calibration()
    values.insert(K(0), calibration)
    graph.push_back(gtsam.PriorFactorCal3Bundler(K(0), calibration, gtsam.noiseModel.Constrained.All(3)))
    camera_keys = [X(i) for i in scene.get_valid_camera_indices()]
    for i in scene.get_valid_camera_indices():
        values.insert(X(i), scene.get_camera(i).pose())
    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    for j, track in enumerate(scene.tracks()):
        values.insert(L(j), track.point3())
        for k in range(track.numberMeasurements()):
            i, uv = track.measurement(k)
            graph.push_back(gtsam.GeneralSFMFactor2Cal3Bundler(uv, noise, X(i), L(j), K(0)))
    _add_graph_gauge(graph, values, camera_keys)
    return graph, values


def _sparse_bearing_graph(number_tracks: int) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
    """Construct twelve cameras with many tracks, each spanning three neighboring cameras."""
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    for i in range(12):
        values.insert(X(i), gtsam.Pose3(gtsam.Rot3(), np.array([i * 0.25, 0.2 * (i % 3), 0.05 * (i % 2)])))
    rng = np.random.default_rng(8)
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), gtsam.noiseModel.Isotropic.Sigma(2, 0.003)
    )
    for j in range(number_tracks):
        point = rng.uniform([-1, -1, 4], [4, 2, 8])
        values.insert(L(j), point)
        for i in range(j % 10, j % 10 + 3):
            direction = values.atPose3(X(i)).transformTo(point) + rng.normal(0, 0.02, 3)
            graph.push_back(gtsam.BearingFactor3D(X(i), L(j), gtsam.Unit3(direction), noise))
    _add_graph_gauge(graph, values, [X(i) for i in range(12)])
    return graph, values


def _camera_vector(values: gtsam.Values, seed: int) -> gtsam.VectorValues:
    """Sample camera increments that satisfy the exact pose and baseline coordinate rows."""
    rng = np.random.default_rng(seed)
    keys = [key for key in values.keys() if gtsam.Symbol(key).chr() == ord("x")]
    anchor = values.atPose3(keys[0])
    farthest = max(keys[1:], key=lambda key: anchor.range(values.atPose3(key)))
    increments = gtsam.VectorValues()
    for key in keys:
        delta = rng.normal(0, 1e-4, 6) if key != keys[0] else np.zeros(6)
        if key == farthest:
            pose = values.atPose3(key)
            direction = pose.rotation().matrix().T @ (pose.translation() - anchor.translation())
            direction /= np.linalg.norm(direction)
            delta[3:] -= direction * (direction @ delta[3:])
        increments.insert(key, delta)
    return increments


@pytest.mark.parametrize("number_tracks", [100, 1000])
def test_camera_marginals_are_compact_sparse_and_preserve_quadratic(number_tracks: int) -> None:
    """Factor count scales with cameras while sparse scopes, offsets, weights, and gauge survive."""
    graph, values = _sparse_bearing_graph(number_tracks)
    original_graph, original_values = graph.clone(), gtsam.Values(values)
    with patch.object(cto, "_add_coordinate_gauge", side_effect=AssertionError("Original gauge replaced")):
        compact = cto.compute_camera_marginal_factors(graph, values)
    camera_keys = [X(i) for i in range(12)]
    reduced = graph.linearize(values).marginal(camera_keys, gtsam.EliminateQR)
    assert set(compact.keyVector()) == set(camera_keys)
    assert compact.size() <= 12 < reduced.size()
    scopes = [compact.at(k).keys() for k in range(compact.size())]
    assert max(map(len, scopes)) < 12
    assert sum(map(len, scopes)) < 12 * 12 / 2
    assert any(gtsam.JacobianFactor(compact.at(k)).isConstrained() for k in range(compact.size()))
    zero = gtsam.VectorValues()
    for key in camera_keys:
        zero.insert(key, np.zeros(6))
    for seed in range(4):
        vector = _camera_vector(values, seed)
        np.testing.assert_allclose(
            compact.error(vector) - compact.error(zero),
            reduced.error(vector) - reduced.error(zero),
            rtol=1e-6,
            atol=1e-7,
        )
    assert graph.equals(original_graph, 1e-12)
    assert values.equals(original_values, 0)


def test_camera_marginal_ordering_is_landmarks_first_then_nested_dissection() -> None:
    """Both stages use multifrontal QR, with METIS computed on the reduced camera graph."""
    graph, values = _sparse_bearing_graph(100)
    eliminate_partial = gtsam.GaussianFactorGraph.eliminatePartialMultifrontal
    eliminate_full = gtsam.GaussianFactorGraph.eliminateMultifrontal
    metis = gtsam.Ordering.MetisGaussianFactorGraph
    calls = []

    def eliminate_landmarks(linear_graph, ordering, method):
        """Check that the first numerical stage eliminates only landmarks using QR."""
        assert {ordering.at(k) for k in range(ordering.size())} == {L(j) for j in range(100)}
        assert method is gtsam.EliminateQR
        calls.append("landmarks")
        return eliminate_partial(linear_graph, ordering, method)

    def order_cameras(camera_graph):
        """Compute nested dissection only after all landmark variables disappear."""
        assert set(camera_graph.keyVector()) == {X(i) for i in range(12)}
        calls.append("metis")
        return metis(camera_graph)

    def eliminate_cameras(camera_graph, ordering, method):
        """Check that camera multifrontal QR uses precisely the METIS ordering."""
        assert ordering.equals(metis(camera_graph), 0)
        assert method is gtsam.EliminateQR
        calls.append("cameras")
        return eliminate_full(camera_graph, ordering, method)

    with (
        patch.object(gtsam.GaussianFactorGraph, "eliminatePartialMultifrontal", eliminate_landmarks),
        patch.object(gtsam.Ordering, "MetisGaussianFactorGraph", order_cameras),
        patch.object(gtsam.GaussianFactorGraph, "eliminateMultifrontal", eliminate_cameras),
    ):
        cto.compute_camera_marginal_factors(graph, values)
    assert calls == ["landmarks", "metis", "cameras"]


def _original_leaf_factors(scene: GtsfmData) -> cto.CameraLeafFactors:
    """Package a compact original-graph marginal for the camera-only tree consumer."""
    graph, values = _original_leaf_graph(scene)
    camera_graph = cto.compute_camera_marginal_factors(graph, values)
    assert camera_graph.size() <= 5
    poses = gtsam.Values()
    metadata = {}
    for i in scene.get_valid_camera_indices():
        poses.insert(X(i), values.atPose3(X(i)))
        info = scene.get_image_info(i)
        metadata[X(i)] = cto.LeafCameraMetadata(info.name, scene.get_camera(i).calibration(), info.shape)
    return cto.CameraLeafFactors(camera_graph, poses, metadata, gauge_fixed=True)


def _assert_lund_camera_information(actual: cto.CameraTreeResult, expected: cto.CameraTreeResult) -> None:
    """Compare final information independently of QR row bases and signs."""
    ordering = gtsam.Ordering([X(i) for i in range(12)])
    actual_a, actual_b = gtsam.GaussianFactorGraph(actual.camera_bayes_tree).jacobian(ordering)
    expected_a, expected_b = gtsam.GaussianFactorGraph(expected.camera_bayes_tree).jacobian(ordering)
    np.testing.assert_allclose(actual_a.T @ actual_a, expected_a.T @ expected_a, rtol=1e-6, atol=1e-4)
    # The constrained Lund system has coefficients up to 1e10; compare linear terms in scaled coordinates.
    scales = np.maximum(np.linalg.norm(expected_a, axis=0), 1)
    np.testing.assert_allclose((actual_a.T @ actual_b) / scales, (expected_a.T @ expected_b) / scales, atol=5e-5)


def test_original_graph_lund_matches_reconstruction_route() -> None:
    """Original graphs, including exact calibration/gauge factors, feed the existing consumer."""
    tree = _overlapping_lund_tree()
    by_path = {node.value[0]: _original_leaf_factors(node.value[1]) for node in tree.leaves()}
    with patch.object(cto, "compute_camera_schur_complement", side_effect=AssertionError("Reconstruction read")):
        result = cto.optimize_camera_factor_tree(tree.map(lambda value: (value[0], by_path.get(value[0]))))
    with pytest.warns(UserWarning, match="original BA gauge"):
        expected = cto.optimize_camera_tree(tree)
    _assert_same_solution(result, expected, compare_factorization=False)
    _assert_lund_camera_information(result, expected)


def test_original_graph_preserves_soft_priors_and_nonzero_offsets() -> None:
    """Original prior targets and measurement weights survive camera marginalization."""
    graph, values = _original_leaf_graph(_synthetic_scene())
    graph.push_back(
        gtsam.PriorFactorPose3(
            X(1), values.atPose3(X(1)).retract(np.full(6, 0.02)), gtsam.noiseModel.Isotropic.Sigma(6, 0.07)
        )
    )
    compact = cto.compute_camera_marginal_factors(graph, values)
    reduced = graph.linearize(values).marginal([X(i) for i in range(3)], gtsam.EliminateQR)
    assert np.linalg.norm(compact.gradientAtZero().at(X(1))) > 1
    for seed in range(4):
        vector = _camera_vector(values, seed)
        zero = gtsam.VectorValues()
        for key in compact.keyVector():
            zero.insert(key, np.zeros(6))
        np.testing.assert_allclose(
            compact.error(vector) - compact.error(zero), reduced.error(vector) - reduced.error(zero), atol=1e-8
        )


def test_explicit_noncontiguous_camera_keys() -> None:
    """Nonconventional local camera keys are supported through explicit key selection."""
    graph, values = _original_leaf_graph(_synthetic_scene())
    keys = [105, 2, 7001]
    mapping = {**{key: key for key in graph.keyVector()}, **{X(i): key for i, key in enumerate(keys)}}
    rekeyed_graph = gtsam.NonlinearFactorGraph()
    for k in range(graph.size()):
        factor = graph.at(k)
        rekeyed_graph.push_back(factor.rekey([mapping[key] for key in factor.keys()]))
    rekeyed_values = gtsam.Values()
    rekeyed_values.insert(K(0), values.atCal3Bundler(K(0)))
    for i, key in enumerate(keys):
        rekeyed_values.insert(key, values.atPose3(X(i)))
    for j in range(30):
        rekeyed_values.insert(L(j), values.atPoint3(L(j)))
    assert set(cto.compute_camera_marginal_factors(rekeyed_graph, rekeyed_values, keys).keyVector()) == set(keys)


@pytest.mark.parametrize(
    "case", ["missing_values", "unknown_key", "duplicate_key", "wrong_type", "no_gauge", "nonfinite", "nonfinite_point"]
)
def test_invalid_original_graph_input(case: str) -> None:
    """Reject invalid original graph inputs and require a supplied full coordinate gauge."""
    graph, values = _original_leaf_graph(_synthetic_scene())
    camera_keys = [X(i) for i in range(3)]
    if case == "missing_values":
        values.erase(L(0))
    elif case == "unknown_key":
        camera_keys = [X(99)]
    elif case == "duplicate_key":
        camera_keys = [X(0), X(0)]
    elif case == "wrong_type":
        camera_keys = [L(0)]
    elif case == "no_gauge":
        graph.remove(graph.size() - 1)
        graph.remove(graph.size() - 2)
    elif case == "nonfinite_point":
        values.update(L(0), np.full(3, np.nan))
    else:
        values.update(X(0), gtsam.Pose3(gtsam.Rot3(), np.full(3, np.nan)))
    with pytest.raises(ValueError):
        cto.compute_camera_marginal_factors(graph, values, camera_keys)


def test_scene_tree_wrapper_warns_and_remains_available() -> None:
    """The geometry-only convenience API directs callers to the original-graph producer."""
    tree = Tree((Path("leaf"), _synthetic_scene()))
    with pytest.warns(UserWarning, match="compute_camera_marginal_factors"):
        assert cto.optimize_camera_tree(tree).scene.number_images() == 3


def test_original_graph_api_does_not_rebuild_or_optimize() -> None:
    """The graph producer must consume the caller's likelihood without reconstruction or BA calls."""
    graph, values = _original_leaf_graph(_synthetic_scene())
    with (
        patch.object(cto, "_linearize_reconstruction", side_effect=AssertionError("Observations rebuilt")),
        patch.object(
            cto, "compute_camera_schur_complement", side_effect=AssertionError("Reconstruction producer called")
        ),
        patch(
            "gtsfm.bundle.bundle_adjustment.BundleAdjustmentOptimizer.run_ba", side_effect=AssertionError("BA called")
        ),
        patch(
            "gtsfm.bundle.bundle_adjustment.BundleAdjustmentOptimizer.run_simple_ba",
            side_effect=AssertionError("BA called"),
        ),
    ):
        assert set(cto.compute_camera_marginal_factors(graph, values).keyVector()) == {X(i) for i in range(3)}
