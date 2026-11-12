"""Numerical and real-reconstruction tests for camera-only hierarchical fusion."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import gtsam
import numpy as np
import pytest
from gtsam.symbol_shorthand import K, L, S, X

from gtsfm import camera_tree_optimizer as cto
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.products import scene_tree
from gtsfm.utils.tree import PreOrderIter, Tree

DATA_ROOT = Path(__file__).resolve().parent / "data"
FIXED_OPTIONS = cto.CameraTreeOptions(optimize_intrinsics=False)


def _synthetic_scene(camera_ids: tuple[int, ...] = (0, 1, 2), degenerate: bool = False) -> GtsfmData:
    """Make a small calibrated scene with noisy observations and known identities."""
    scene = GtsfmData(len(camera_ids))
    centers = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.1]), np.array([0.2, 0.8, 0.0])]
    calibration = gtsam.Cal3Bundler(400.0, 0.0, 0.0, 320.0, 240.0)
    for i, camera_id in enumerate(camera_ids):
        scene.add_camera(i, gtsam.PinholeCameraCal3Bundler(gtsam.Pose3(gtsam.Rot3(), centers[i]), calibration))
        scene.set_image_info(i, name=f"image_{camera_id:02}.jpg", shape=(480, 640))
    rng = np.random.default_rng(8)
    for _ in range(30):
        point = np.array([0.1, 0.2, 5.0]) if degenerate else rng.uniform([-1, -1, 4], [2, 2, 8])
        track = gtsam.SfmTrack(point)
        for i in range(len(camera_ids)):
            uv = scene.get_camera(i).project(point) + rng.normal(0, 0.2, 2)
            track.addMeasurement(i, uv)
        scene.add_track(track)
    return scene


def _single_leaf(scene: GtsfmData) -> cto._LeafInput:
    """Resolve identities for a one-leaf numerical test."""
    return cto._leaf_inputs(Tree((Path("leaf"), cto.compute_camera_schur_complement(scene, FIXED_OPTIONS))))[0]


def _vector_values(keys: list[int], delta: np.ndarray) -> gtsam.VectorValues:
    """Pack six-dimensional camera increments in a GTSAM VectorValues."""
    values = gtsam.VectorValues()
    for i, key in enumerate(keys):
        values.insert(key, delta[6 * i : 6 * (i + 1)])
    return values


def test_landmark_elimination_and_bayes_tree_preserve_quadratic() -> None:
    """Check the Schur objective against explicit minimization of all point increments."""
    scene = _synthetic_scene()
    leaf = _single_leaf(scene)
    options = FIXED_OPTIONS
    graph = cto._linearize_reconstruction(scene, options)
    estimated_camera_poses = leaf.data.estimated_camera_poses
    point_ordering = gtsam.Ordering([L(j) for j in range(scene.number_tracks())])
    camera_keys = list(estimated_camera_poses.keys())
    full_ordering = gtsam.Ordering()
    for key in camera_keys:
        full_ordering.push_back(key)
    for j in range(scene.number_tracks()):
        full_ordering.push_back(L(j))
    matrix, rhs = graph.jacobian(full_ordering)
    camera_matrix = matrix[:, : 6 * len(camera_keys)]
    point_matrix = matrix[:, 6 * len(camera_keys) :]
    _, reduced = graph.eliminatePartialMultifrontal(point_ordering, gtsam.EliminateQR)
    assert set(reduced.keyVector()) == set(camera_keys)
    assert leaf.data.camera_graph.equals(reduced, 1e-8)
    summary = cto._prepare_leaf(leaf, options)
    tree_graph = gtsam.GaussianFactorGraph(summary.camera_bayes_tree)
    zero = _vector_values(camera_keys, np.zeros(6 * len(camera_keys)))
    zero_points = np.linalg.lstsq(point_matrix, rhs, rcond=None)[0]
    zero_profile = 0.5 * np.linalg.norm(point_matrix @ zero_points - rhs) ** 2
    assert np.isfinite(zero_profile)
    rng = np.random.default_rng(3)
    anchor = estimated_camera_poses.atPose3(camera_keys[0])
    farthest = max(
        range(1, len(camera_keys)),
        key=lambda i: np.linalg.norm(
            estimated_camera_poses.atPose3(camera_keys[i]).translation() - anchor.translation()
        ),
    )
    pose = estimated_camera_poses.atPose3(camera_keys[farthest])
    direction = pose.rotation().matrix().T @ (pose.translation() - anchor.translation())
    direction /= np.linalg.norm(direction)
    for _ in range(4):
        delta = rng.normal(0, 0.01, 6 * len(camera_keys))
        delta[:6] = 0
        translation = delta[6 * farthest + 3 : 6 * farthest + 6]
        translation -= direction * (direction @ translation)
        vector = _vector_values(camera_keys, delta)
        point_delta = np.linalg.lstsq(point_matrix, rhs - camera_matrix @ delta, rcond=None)[0]
        profile = 0.5 * np.linalg.norm(camera_matrix @ delta + point_matrix @ point_delta - rhs) ** 2
        expected_difference = profile - zero_profile
        assert np.isfinite(expected_difference)
        np.testing.assert_allclose(
            reduced.error(vector) - reduced.error(zero), expected_difference, rtol=1e-9, atol=1e-8
        )
        np.testing.assert_allclose(
            tree_graph.error(_vector_values([X(i) for i in camera_keys], delta))
            - tree_graph.error(_vector_values([X(i) for i in camera_keys], np.zeros_like(delta))),
            expected_difference,
            rtol=1e-9,
            atol=1e-8,
        )


def test_similarity_factor_analytic_jacobians() -> None:
    """Check pose and similarity derivatives away from identity and at nonunit scale."""
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(9, 12))
    rhs = rng.normal(size=9)
    gaussian = gtsam.JacobianFactor(X(0), matrix[:, :6], X(1), matrix[:, 6:], rhs, gtsam.noiseModel.Unit.Create(9))
    estimated_camera_poses = gtsam.Values()
    estimated_camera_poses.insert(X(0), gtsam.Pose3.Expmap(rng.normal(0, 0.2, 6)))
    estimated_camera_poses.insert(X(1), gtsam.Pose3.Expmap(rng.normal(0, 0.2, 6)))
    factor = cto._lift_conditional(gaussian, estimated_camera_poses, S(0))
    values = gtsam.Values()
    similarity = gtsam.Similarity3(gtsam.Rot3.RzRyRx(0.2, -0.3, 0.1), np.array([1.0, -2.0, 0.4]), 2.3)
    values.insert(S(0), similarity)
    for key in estimated_camera_poses.keys():
        pose = estimated_camera_poses.atPose3(key).retract(rng.normal(0, 0.1, 6))
        values.insert(key, similarity.inverse().transformFrom(pose))
    linear = gtsam.JacobianFactor(factor.linearize(values))
    column = 0
    for key in linear.keys():
        is_similarity = key == S(0)
        value = values.atSimilarity3(key) if is_similarity else values.atPose3(key)
        dimension = 7 if is_similarity else 6
        numerical = np.zeros((9, dimension))
        for k in range(dimension):
            step = np.zeros(dimension)
            step[k] = 1e-6
            plus, minus = gtsam.Values(values), gtsam.Values(values)
            plus.update(key, value.retract(step))
            minus.update(key, value.retract(-step))
            numerical[:, k] = (factor.unwhitenedError(plus) - factor.unwhitenedError(minus)) / (2e-6)
        np.testing.assert_allclose(linear.getA()[:, column : column + dimension], numerical, atol=2e-7, rtol=2e-6)
        column += dimension


def _overlapping_lund_tree() -> scene_tree.SceneTree:
    """Reuse Lund's topology with overlapping, independently owned source tracks."""
    topology = scene_tree.read_colmap(str(DATA_ROOT / "lund_door_binary"))
    assert topology is not None
    source = GtsfmData.read_colmap(str(DATA_ROOT / "set1_lund_door" / "colmap_ground_truth"))
    groups = [list(range(0, 5)), list(range(2, 7)), list(range(5, 10)), list(range(7, 12))]
    scenes = [GtsfmData(len(group)) for group in groups]
    for scene, group in zip(scenes, groups):
        for local_i, global_i in enumerate(group):
            scene.add_camera(local_i, source.get_camera(global_i))
            info = source.get_image_info(global_i)
            scene.set_image_info(local_i, name=info.name, shape=info.shape)
    for j, track in enumerate(source.tracks()):
        observations = [track.measurement(k) for k in range(track.numberMeasurements())]
        cameras = {i for i, _ in observations}
        counts = [len(cameras.intersection(group)) for group in groups]
        best = max(counts)
        if best < 2:
            continue
        candidates = [index for index, count in enumerate(counts) if count == best]
        owner = candidates[j % len(candidates)]
        selected = gtsam.SfmTrack(track.point3())
        for i, uv in observations:
            if i in groups[owner]:
                selected.addMeasurement(groups[owner].index(i), uv)
        scenes[owner].add_track(selected)
    by_path = {node.value[0]: scene for node, scene in zip(topology.leaves(), scenes)}
    return topology.map(lambda value: (value[0], by_path.get(value[0])))


@pytest.fixture(scope="module")
def lund_tree() -> scene_tree.SceneTree:
    """Load the overlapping real reconstruction fixture once."""
    return _overlapping_lund_tree()


@pytest.fixture(scope="module", params=[False, True], ids=["fixed", "intrinsics"])
def lund_options(request: pytest.FixtureRequest) -> cto.CameraTreeOptions:
    """Exercise both calibration modes on the same real reconstruction."""
    return cto.CameraTreeOptions(optimize_intrinsics=request.param)


@pytest.fixture(scope="module")
def lund_result(lund_tree: scene_tree.SceneTree, lund_options: cto.CameraTreeOptions) -> cto.CameraTreeResult:
    """Solve the real reconstruction while prohibiting any BA optimizer calls."""
    with (
        patch(
            "gtsfm.bundle.bundle_adjustment.BundleAdjustmentOptimizer.run_ba", side_effect=AssertionError("BA called")
        ),
        patch(
            "gtsfm.bundle.bundle_adjustment.BundleAdjustmentOptimizer.run_simple_ba",
            side_effect=AssertionError("BA called"),
        ),
    ):
        return cto.optimize_camera_tree(lund_tree, lund_options)


def test_lund_camera_only_merge(lund_tree: scene_tree.SceneTree, lund_result: cto.CameraTreeResult) -> None:
    """Check actual reconstruction optimization, factor ownership, and camera-only outputs."""
    result = lund_result
    assert result.scene.get_valid_camera_indices() == list(range(12))
    assert result.scene.number_tracks() == 0
    assert result.final_error < result.initial_error
    assert result.iterations < 100
    assert np.isfinite(result.final_error)
    assert len(result.leaf_to_global) == 4
    assert len(list(PreOrderIter(result.factor_tree))) == len(list(PreOrderIter(lund_tree))) == 7
    for source_node, result_node in zip(PreOrderIter(lund_tree), PreOrderIter(result.factor_tree)):
        assert len(source_node.children) == len(result_node.children)
        assert result_node.value.graph.size() == sum(len(leaf.factors) for leaf in result_node.value.leaves)
    for leaf in result.factor_tree.value.leaves:
        keys = gtsam.GaussianFactorGraph(leaf.camera_bayes_tree).keyVector()
        assert set(keys) == {X(i) for i in leaf.local_to_global.values()} | set(leaf.estimated_calibrations.keys())
    expected_keys = {X(i) for i in range(12)}
    if result.factor_tree.value.leaves[0].estimated_calibrations.size():
        expected_keys |= {K(i) for i in range(12)}
    assert set(gtsam.GaussianFactorGraph(result.camera_bayes_tree).keyVector()) == expected_keys
    for i in range(12):
        assert result.scene.get_image_info(i).name == f"DSC_{i + 1:04}.JPG"
        assert np.all(np.isfinite(result.scene.get_camera(i).pose().matrix()))
        assert np.all(np.isfinite(result.scene.get_camera(i).calibration().vector()))


def test_lund_independent_leaf_frame_invariance(
    lund_tree: scene_tree.SceneTree, lund_result: cto.CameraTreeResult, lund_options: cto.CameraTreeOptions
) -> None:
    """Large independent changes of leaf gauge must preserve the fused camera solution."""
    transforms = [
        gtsam.Similarity3(gtsam.Rot3.RzRyRx(0.2, -0.1, 0.3), np.array([2.0, -1.0, 0.4]), 1.4),
        gtsam.Similarity3(gtsam.Rot3.RzRyRx(-0.3, 0.2, 0.1), np.array([-1.0, 2.0, 0.5]), 0.6),
        gtsam.Similarity3(gtsam.Rot3.RzRyRx(0.1, 0.3, -0.2), np.array([0.5, 0.2, -1.0]), 2.1),
        gtsam.Similarity3(gtsam.Rot3.RzRyRx(-0.2, -0.1, 0.2), np.array([1.0, -0.5, 2.0]), 0.8),
    ]
    by_path = {node.value[0]: transform for node, transform in zip(lund_tree.leaves(), transforms)}
    reframed = lund_tree.map(
        lambda value: (value[0], value[1].transform_with_sim3(by_path[value[0]]) if value[1] is not None else None)
    )
    result = cto.optimize_camera_tree(reframed, lund_options)
    reframed_to_original = transforms[0].inverse()
    for i in range(12):
        expected = lund_result.scene.get_camera(i).pose()
        actual = reframed_to_original.transformFrom(result.scene.get_camera(i).pose())
        np.testing.assert_allclose(expected.localCoordinates(actual), np.zeros(6), atol=2e-5)
        np.testing.assert_allclose(
            result.scene.get_camera(i).calibration().vector(),
            lund_result.scene.get_camera(i).calibration().vector(),
            rtol=1e-6,
            atol=2e-5,
        )
    np.testing.assert_allclose(result.final_error, lund_result.final_error, rtol=1e-6, atol=1e-6)


def test_perturbed_initialization_and_ignored_internal_scenes(
    lund_tree: scene_tree.SceneTree, lund_options: cto.CameraTreeOptions
) -> None:
    """The root solve must correct perturbations without accessing internal scene payloads."""
    initialize = cto._initialize

    def perturb(leaves: tuple[cto._LeafInput, ...]) -> gtsam.Values:
        """Perturb camera estimates but keep the first leaf's similarity gauge feasible."""
        values = initialize(leaves)
        rng = np.random.default_rng(2)
        for key in values.keys():
            if gtsam.Symbol(key).chr() == ord("x"):
                values.update(key, values.atPose3(key).retract(rng.normal(0, 1e-4, 6)))
            elif gtsam.Symbol(key).chr() == ord("k"):
                values.update(key, values.atCal3Bundler(key).retract(np.array([0.1, 1e-5, -1e-5])))
            elif key != S(0):
                values.update(key, values.atSimilarity3(key).retract(rng.normal(0, 1e-4, 7)))
        return values

    poisoned = lund_tree.map(lambda value: (value[0], value[1] if value[1] is not None else object()))
    with patch.object(cto, "_initialize", side_effect=perturb):
        result = cto.optimize_camera_tree(poisoned, lund_options)
    assert result.final_error < result.initial_error
    assert result.scene.number_images() == 12


def test_original_lund_leaves_are_unobservable() -> None:
    """The unchanged fixture has four disjoint camera sets and cannot determine relative scales."""
    tree = scene_tree.read_colmap(str(DATA_ROOT / "lund_door_binary"))
    assert tree is not None
    with pytest.raises(ValueError, match="Disconnected or insufficient camera overlap.*C_1/C_1_2"):
        cto.optimize_camera_tree(tree, FIXED_OPTIONS)


def test_one_shared_camera_is_insufficient() -> None:
    """A single shared camera does not fix the relative leaf scale."""
    tree = Tree(
        (Path("root"), None),
        (Tree((Path("a"), _synthetic_scene((0, 1, 2)))), Tree((Path("b"), _synthetic_scene((2, 3, 4))))),
    )
    with pytest.raises(ValueError, match="two shared cameras"):
        cto.optimize_camera_tree(tree, FIXED_OPTIONS)


def test_degenerate_baseline() -> None:
    """Coincident centers must fail before similarity initialization or point elimination."""
    scene = _synthetic_scene()
    for i in scene.get_valid_camera_indices():
        scene.update_camera(i, gtsam.PinholeCameraCal3Bundler(gtsam.Pose3(), scene.get_camera(i).calibration()))
    with pytest.raises(ValueError, match="degenerate camera baseline"):
        cto.optimize_camera_tree(Tree((Path("degenerate"), scene)))


def test_missing_camera_identity() -> None:
    """Locally renumbered cameras cannot be matched without their filenames."""
    scene = _synthetic_scene()
    scene.get_image_info(0).name = None
    with pytest.raises(ValueError, match="missing or duplicate camera filename"):
        cto.optimize_camera_tree(Tree((Path("unnamed"), scene)))


def test_insufficient_geometric_rank() -> None:
    """Repeated observations of one 3D point cannot constrain a camera reconstruction."""
    with pytest.raises(ValueError, match="Insufficient geometric rank"):
        cto.optimize_camera_tree(Tree((Path("low_rank"), _synthetic_scene(degenerate=True))))


@pytest.mark.parametrize(
    "kwargs",
    [{"measurement_sigma": 0}, {"max_iterations": 0}, {"rank_tolerance": float("nan")}, {"optimize_intrinsics": 1}],
)
def test_invalid_options(kwargs: dict) -> None:
    """Reject invalid numerical options at construction."""
    with pytest.raises(ValueError):
        cto.CameraTreeOptions(**kwargs)


@pytest.fixture(scope="module")
def lund_camera_factors(lund_tree: scene_tree.SceneTree, lund_options: cto.CameraTreeOptions) -> cto.CameraFactorTree:
    """Construct the external input using only the public Schur producer."""
    by_path = {
        node.value[0]: cto.compute_camera_schur_complement(node.value[1], lund_options) for node in lund_tree.leaves()
    }
    return lund_tree.map(lambda value: (value[0], by_path.get(value[0])))


def _assert_same_solution(actual: cto.CameraTreeResult, expected: cto.CameraTreeResult) -> None:
    """Compare reconstructed cameras, leaf similarities, objectives, and final quadratic.

    Args:
        actual: Result from the tested input route.
        expected: Baseline result in the same global gauge.
    """
    assert actual.scene.get_valid_camera_indices() == expected.scene.get_valid_camera_indices()
    for i in actual.scene.get_valid_camera_indices():
        assert actual.scene.get_camera(i).equals(expected.scene.get_camera(i), 1e-7)
        assert actual.scene.get_image_info(i) == expected.scene.get_image_info(i)
    for path in expected.leaf_to_global:
        assert actual.leaf_to_global[path].equals(expected.leaf_to_global[path], 1e-7)
    np.testing.assert_allclose(
        [actual.initial_error, actual.final_error], [expected.initial_error, expected.final_error], rtol=1e-7, atol=1e-7
    )
    a = gtsam.GaussianFactorGraph(actual.camera_bayes_tree)
    b = gtsam.GaussianFactorGraph(expected.camera_bayes_tree)
    assert a.equals(b, 1e-5)
    assert [len(n.children) for n in PreOrderIter(actual.factor_tree)] == [
        len(n.children) for n in PreOrderIter(expected.factor_tree)
    ]


def test_external_camera_only_execution(
    lund_camera_factors: cto.CameraFactorTree, lund_result: cto.CameraTreeResult
) -> None:
    """The external route must not reconstruct, relinearize observations, or eliminate landmarks."""
    marginal = gtsam.GaussianFactorGraph.marginal
    calls = []

    def camera_marginal(graph, keys, *args):
        """Only allow the final marginalization of global camera/similarity variables."""
        assert all(gtsam.Symbol(key).chr() in (ord("x"), ord("s"), ord("k")) for key in graph.keyVector())
        calls.append(list(keys))
        return marginal(graph, keys, *args)

    with (
        patch.object(cto, "compute_camera_schur_complement", side_effect=AssertionError("Producer called")),
        patch.object(cto, "_linearize_reconstruction", side_effect=AssertionError("Observations relinearized")),
        patch.object(GtsfmData, "tracks", side_effect=AssertionError("Tracks accessed")),
        patch(
            "gtsfm.bundle.bundle_adjustment.BundleAdjustmentOptimizer.run_ba", side_effect=AssertionError("BA called")
        ),
        patch(
            "gtsfm.bundle.bundle_adjustment.BundleAdjustmentOptimizer.run_simple_ba",
            side_effect=AssertionError("BA called"),
        ),
        patch.object(gtsam.GaussianFactorGraph, "marginal", camera_marginal),
    ):
        result = cto.optimize_camera_factor_tree(lund_camera_factors, cto.CameraTreeOptions(measurement_sigma=9.0))
    expected_keys = [X(i) for i in range(12)]
    if lund_result.factor_tree.value.leaves[0].estimated_calibrations.size():
        expected_keys += [K(i) for i in range(12)]
    assert calls == [expected_keys]
    _assert_same_solution(result, lund_result)


def _gauge_fixed(data: cto.CameraLeafFactors) -> cto.CameraLeafFactors:
    """Add the coordinate gauge externally without modifying the original graph.

    Args:
        data: Ungauged camera quadratic.

    Returns:
        Independently anchored camera input.
    """
    graph = data.camera_graph.clone()
    cto._add_coordinate_gauge(graph, data.estimated_camera_poses)
    return replace(data, camera_graph=graph, gauge_fixed=True)


@pytest.mark.parametrize("fixed_count", [2, 4])
def test_external_gauge_modes(
    lund_camera_factors: cto.CameraFactorTree, lund_result: cto.CameraTreeResult, fixed_count: int
) -> None:
    """Accept mixed or fully anchored leaves without duplicating their coordinate constraints."""
    by_path = {
        node.value[0]: _gauge_fixed(node.value[1]) if i < fixed_count else node.value[1]
        for i, node in enumerate(lund_camera_factors.leaves())
    }
    tree = lund_camera_factors.map(lambda value: (value[0], by_path.get(value[0])))
    with patch.object(cto, "_add_coordinate_gauge", wraps=cto._add_coordinate_gauge) as add_gauge:
        result = cto.optimize_camera_factor_tree(tree)
    assert add_gauge.call_count == 4 - fixed_count
    _assert_same_solution(result, lund_result)


def _rekey_camera_input(data: cto.CameraLeafFactors, mapping: dict[int, int]) -> cto.CameraLeafFactors:
    """Represent an external producer's independent camera key space.

    Args:
        data: Camera factors to copy.
        mapping: Complete old-to-new key mapping.

    Returns:
        Equivalent input under arbitrary local camera keys.
    """
    pose_mapping = dict(mapping)
    mapping = {**mapping, **{key: 2000 + i for i, key in enumerate(data.calibration_keys.values())}}
    graph = gtsam.GaussianFactorGraph()
    for k in range(data.camera_graph.size()):
        factor = data.camera_graph.at(k)
        graph.push_back(
            gtsam.LinearContainerFactor(factor.clone()).rekey([mapping[key] for key in factor.keys()]).factor()
        )
    poses = gtsam.Values()
    for key, new_key in pose_mapping.items():
        poses.insert(new_key, data.estimated_camera_poses.atPose3(key))
    return replace(
        data,
        camera_graph=graph,
        estimated_camera_poses=poses,
        camera_metadata={mapping[key]: info for key, info in data.camera_metadata.items()},
        calibration_keys={mapping[key]: mapping[cal_key] for key, cal_key in data.calibration_keys.items()},
    )


def test_independent_keys_and_input_immutability(
    lund_camera_factors: cto.CameraFactorTree, lund_result: cto.CameraTreeResult
) -> None:
    """Reused, noncontiguous keys must resolve by filename without mutating caller objects."""
    # Deliberately reuse both camera-like and similarity-like keys with a nonmonotonic mapping.
    keys = [S(0), 8, X(90), 1003, 2]
    by_path = {n.value[0]: _rekey_camera_input(n.value[1], dict(enumerate(keys))) for n in lund_camera_factors.leaves()}
    snapshots = {
        p: (
            d.camera_graph.clone(),
            gtsam.Values(d.estimated_camera_poses),
            dict(d.camera_metadata),
            dict(d.calibration_keys),
        )
        for p, d in by_path.items()
    }
    tree = lund_camera_factors.map(lambda value: (value[0], by_path.get(value[0], object())))
    result = cto.optimize_camera_factor_tree(tree)
    _assert_same_solution(result, lund_result)
    for leaf in result.factor_tree.value.leaves:
        data = by_path[leaf.path]
        graph, poses, metadata, calibration_keys = snapshots[leaf.path]
        assert data.camera_graph.equals(graph, 0)
        assert data.estimated_camera_poses.equals(poses, 0)
        assert data.camera_metadata == metadata
        assert data.calibration_keys == calibration_keys
        assert set(leaf.local_to_global) == set(keys)
        assert set(leaf.estimated_camera_poses.keys()) == {X(i) for i in leaf.local_to_global.values()}


def test_public_producer_is_ungauged_and_does_not_run_ba() -> None:
    """Expose the full reduced linear term without adding gauge constraints or running BA."""
    scene = _synthetic_scene()
    with (
        patch(
            "gtsfm.bundle.bundle_adjustment.BundleAdjustmentOptimizer.run_ba", side_effect=AssertionError("BA called")
        ),
        patch(
            "gtsfm.bundle.bundle_adjustment.BundleAdjustmentOptimizer.run_simple_ba",
            side_effect=AssertionError("BA called"),
        ),
        patch.object(cto, "_add_coordinate_gauge", side_effect=AssertionError("Gauge added")),
    ):
        data = cto.compute_camera_schur_complement(scene, FIXED_OPTIONS)
    assert data.gauge_fixed is False
    assert (
        set(data.camera_graph.keyVector())
        == set(data.estimated_camera_poses.keys())
        == set(data.camera_metadata)
        == {0, 1, 2}
    )
    a, b = data.camera_graph.jacobian()
    assert np.linalg.norm(a.T @ b) > 1
    assert np.linalg.matrix_rank(a, tol=1e-7) == 11
    assert all(
        not gtsam.JacobianFactor(data.camera_graph.at(k)).isConstrained() for k in range(data.camera_graph.size())
    )


@pytest.fixture(params=[False, True], ids=["fixed", "intrinsics"])
def synthetic_camera_factors(request: pytest.FixtureRequest) -> cto.CameraLeafFactors:
    """Create a compact camera-only likelihood for invalid-input tests."""
    if request.param:
        return cto.compute_camera_schur_complement(_self_calibration_scene())
    return cto.compute_camera_schur_complement(_synthetic_scene(), FIXED_OPTIONS)


@pytest.mark.parametrize("fixed", [False, True])
def test_incorrect_gauge_declaration(synthetic_camera_factors: cto.CameraLeafFactors, fixed: bool) -> None:
    """Reject an ungauged graph declared fixed and a fixed graph declared ungauged."""
    data = synthetic_camera_factors if fixed else _gauge_fixed(synthetic_camera_factors)
    with pytest.raises(ValueError, match="Leaf invalid.*gauge_fixed"):
        cto.optimize_camera_factor_tree(Tree((Path("invalid"), replace(data, gauge_fixed=fixed))))


@pytest.mark.parametrize(
    "case",
    [
        "missing_metadata",
        "graph_keys",
        "pose_type",
        "dimensions",
        "mixed_dimensions",
        "nonfinite",
        "calibration",
        "filename",
    ],
)
def test_invalid_external_camera_input(synthetic_camera_factors: cto.CameraLeafFactors, case: str) -> None:
    """Validate identities and block sizes before passing malformed external graphs to the solver."""
    data = synthetic_camera_factors
    if case == "missing_metadata":
        data = replace(data, camera_metadata={k: v for k, v in data.camera_metadata.items() if k != 0})
    elif case == "graph_keys":
        graph = data.camera_graph.clone()
        graph.push_back(gtsam.JacobianFactor(99, np.eye(6), np.zeros(6), gtsam.noiseModel.Unit.Create(6)))
        data = replace(data, camera_graph=graph)
    elif case == "pose_type":
        poses = gtsam.Values()
        for key in data.estimated_camera_poses.keys():
            poses.insert(key, np.zeros(6) if key == 0 else data.estimated_camera_poses.atPose3(key))
        data = replace(data, estimated_camera_poses=poses)
    elif case in ("dimensions", "mixed_dimensions", "nonfinite"):
        graph = gtsam.GaussianFactorGraph()
        dimensions = [3, 9, 6] if case == "mixed_dimensions" else [3, 3, 3] if case == "dimensions" else [6, 6, 6]
        for key, dimension in enumerate(dimensions):
            rhs = np.full(dimension, np.nan if case == "nonfinite" else 0.0)
            graph.push_back(gtsam.JacobianFactor(key, np.eye(dimension), rhs, gtsam.noiseModel.Unit.Create(dimension)))
        data = replace(data, camera_graph=graph)
    else:
        metadata = dict(data.camera_metadata)
        metadata[0] = (
            replace(metadata[0], calibration=gtsam.Cal3_S2())
            if case == "calibration"
            else replace(metadata[0], filename="")
        )
        data = replace(data, camera_metadata=metadata)
    with pytest.raises(ValueError, match="Leaf invalid"):
        cto.optimize_camera_factor_tree(Tree((Path("invalid"), data)))


def test_external_hessian_factors(synthetic_camera_factors: cto.CameraLeafFactors) -> None:
    """Accept a producer's Hessian representation without discarding its linear term."""
    data = synthetic_camera_factors
    graph = gtsam.GaussianFactorGraph()
    graph.push_back(gtsam.HessianFactor(data.camera_graph))
    result = cto.optimize_camera_factor_tree(Tree((Path("leaf"), replace(data, camera_graph=graph))))
    expected = cto.optimize_camera_factor_tree(Tree((Path("leaf"), data)))
    _assert_same_solution(result, expected)

    with pytest.raises(ValueError, match="Leaf leaf.*gauge_fixed"):
        cto.optimize_camera_factor_tree(Tree((Path("leaf"), replace(data, camera_graph=graph, gauge_fixed=True))))


def _self_calibration_scene() -> GtsfmData:
    """Create nonparallel cameras with observable intrinsics and nonzero residuals."""
    scene = GtsfmData(4)
    rng = np.random.default_rng(81)
    for i in range(4):
        pose = gtsam.Pose3(gtsam.Rot3.RzRyRx(*rng.normal(0, 0.15, 3)), rng.uniform(-1, 1, 3))
        calibration = gtsam.Cal3Bundler(400 + 20 * i, 0.01, -0.005, 320, 240)
        scene.add_camera(i, gtsam.PinholeCameraCal3Bundler(pose, calibration))
        scene.set_image_info(i, name=f"self_cal_{i}.jpg", shape=(480, 640))
    for _ in range(60):
        point = rng.uniform([-2, -2, 4], [2, 2, 8])
        track = gtsam.SfmTrack(point)
        for i in range(4):
            track.addMeasurement(i, scene.get_camera(i).project(point) + rng.normal(0, 0.02, 2))
        scene.add_track(track)
    return scene


def test_joint_schur_preserves_pose_intrinsics_quadratic() -> None:
    """Profile out only points and retain the full mixed camera quadratic and offsets."""
    scene = _self_calibration_scene()
    data = cto.compute_camera_schur_complement(scene)
    pose_keys = list(data.estimated_camera_poses.keys())
    calibration_keys = [data.calibration_keys[key] for key in pose_keys]
    retained = pose_keys + calibration_keys
    point_keys = [L(j) for j in range(scene.number_tracks())]
    graph = cto._linearize_reconstruction(scene, cto.CameraTreeOptions())
    matrix, rhs = graph.jacobian(gtsam.Ordering(retained + point_keys))
    dimension = 9 * len(pose_keys)
    camera_matrix, point_matrix = matrix[:, :dimension], matrix[:, dimension:]
    reduced_a, reduced_b = data.camera_graph.jacobian(gtsam.Ordering(retained))
    assert set(data.camera_graph.keyVector()) == set(retained)
    assert np.linalg.norm(reduced_a[:, : 6 * len(pose_keys)].T @ reduced_a[:, 6 * len(pose_keys) :]) > 1
    assert np.linalg.norm(reduced_a.T @ reduced_b) > 0.01
    zero_points = np.linalg.lstsq(point_matrix, rhs, rcond=None)[0]
    baseline = np.linalg.norm(point_matrix @ zero_points - rhs) ** 2
    assert np.isfinite(baseline)
    leaf = cto._leaf_inputs(Tree((Path("leaf"), data)))[0]
    summary = cto._prepare_leaf(leaf, cto.CameraTreeOptions())
    bayes_graph = gtsam.GaussianFactorGraph(summary.camera_bayes_tree)
    global_keys = [X(leaf.local_to_global[key]) for key in pose_keys] + [
        K(leaf.local_to_global[key]) for key in pose_keys
    ]
    sizes = [6] * len(pose_keys) + [3] * len(calibration_keys)
    zero = gtsam.VectorValues()
    for key, size in zip(global_keys, sizes):
        zero.insert(key, np.zeros(size))
    anchor = data.estimated_camera_poses.atPose3(pose_keys[0])
    farthest = max(
        pose_keys[1:],
        key=lambda key: np.linalg.norm(data.estimated_camera_poses.atPose3(key).translation() - anchor.translation()),
    )
    farthest_pose = data.estimated_camera_poses.atPose3(farthest)
    direction = farthest_pose.rotation().matrix().T @ (farthest_pose.translation() - anchor.translation())
    direction /= np.linalg.norm(direction)
    rng = np.random.default_rng(73)
    for _ in range(4):
        delta = rng.normal(0, 0.001, dimension)
        delta[:6] = 0
        offset = 6 * pose_keys.index(farthest) + 3
        delta[offset : offset + 3] -= direction * (direction @ delta[offset : offset + 3])
        points = np.linalg.lstsq(point_matrix, rhs - camera_matrix @ delta, rcond=None)[0]
        profiled = np.linalg.norm(camera_matrix @ delta + point_matrix @ points - rhs) ** 2
        reduced = np.linalg.norm(reduced_a @ delta - reduced_b) ** 2 - np.linalg.norm(reduced_b) ** 2
        assert np.all(np.isfinite([profiled, reduced]))
        np.testing.assert_allclose(reduced, profiled - baseline, rtol=1e-9, atol=1e-9)
        vector = gtsam.VectorValues()
        offset = 0
        for key, size in zip(global_keys, sizes):
            vector.insert(key, delta[offset : offset + size])
            offset += size
        np.testing.assert_allclose(
            2 * (bayes_graph.error(vector) - bayes_graph.error(zero)), reduced, rtol=1e-9, atol=1e-9
        )


@pytest.mark.parametrize("calibration_only", [False, True])
def test_intrinsics_factor_analytic_jacobians(calibration_only: bool) -> None:
    """Check mixed block offsets and calibration-only conditionals away from the leaf estimate."""
    rng = np.random.default_rng(29)
    rows = 15
    matrix = rng.normal(size=(rows, 12))
    rhs = rng.normal(size=rows)
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.linspace(0.5, 2, rows))
    gaussian = (
        gtsam.JacobianFactor(K(0), matrix[:, :3], rhs, noise)
        if calibration_only
        else gtsam.JacobianFactor(K(0), matrix[:, :3], X(0), matrix[:, 3:9], K(1), matrix[:, 9:], rhs, noise)
    )
    poses, calibrations, values = gtsam.Values(), gtsam.Values(), gtsam.Values()
    similarity = gtsam.Similarity3(gtsam.Rot3.RzRyRx(0.3, -0.2, 0.1), np.array([1, -2, 0.5]), 1.7)
    values.insert(S(0), similarity)
    for i in range(2):
        calibration = gtsam.Cal3Bundler(400 + 20 * i, 0.01, -0.005, 320, 240)
        calibrations.insert(K(i), calibration)
        values.insert(K(i), calibration.retract(np.array([3, 0.001, -0.002])))
    pose = gtsam.Pose3.Expmap(rng.normal(0, 0.2, 6))
    poses.insert(X(0), pose)
    values.insert(X(0), similarity.inverse().transformFrom(pose.retract(rng.normal(0, 0.1, 6))))
    factor = cto._lift_conditional(gaussian, poses, S(0), calibrations)
    linear = gtsam.JacobianFactor(factor.linearize(values))
    assert factor.noiseModel().equals(noise, 0)
    delta = gtsam.VectorValues()
    for key in gaussian.keys():
        delta.insert(
            key,
            (
                poses.atPose3(key).localCoordinates(similarity.transformFrom(values.atPose3(key)))
                if poses.exists(key)
                else calibrations.atCal3Bundler(key).localCoordinates(values.atCal3Bundler(key))
            ),
        )
    np.testing.assert_allclose(factor.error(values), gaussian.error(delta), rtol=1e-12)
    column = 0
    for key in linear.keys():
        if key == S(0):
            value, dimension = values.atSimilarity3(key), 7
        elif key == X(0):
            value, dimension = values.atPose3(key), 6
        else:
            value, dimension = values.atCal3Bundler(key), 3
        numerical = np.zeros((rows, dimension))
        for k in range(dimension):
            step = np.zeros(dimension)
            step[k] = 1e-5
            plus, minus = gtsam.Values(values), gtsam.Values(values)
            plus.update(key, value.retract(step))
            minus.update(key, value.retract(-step))
            numerical[:, k] = (factor.whitenedError(plus) - factor.whitenedError(minus)) / (2e-5)
        np.testing.assert_allclose(linear.getA()[:, column : column + dimension], numerical, atol=2e-7, rtol=2e-6)
        if calibration_only and key == S(0):
            np.testing.assert_array_equal(numerical, 0)
        column += dimension


def test_shared_intrinsics_are_optimized_and_exported() -> None:
    """Fuse independent calibration estimates for each filename instead of copying the first."""
    scene = _self_calibration_scene()
    data = cto.compute_camera_schur_complement(scene)
    # Independent external likelihoods with exact pose coordinates and known calibration optima.
    leaves = []
    for index, change in enumerate((0.0, 4.0)):
        graph = gtsam.GaussianFactorGraph()
        metadata = {}
        for key, info in data.camera_metadata.items():
            graph.push_back(
                gtsam.NonlinearEqualityPose3(key, data.estimated_camera_poses.atPose3(key)).linearize(
                    data.estimated_camera_poses
                )
            )
            graph.push_back(
                gtsam.JacobianFactor(
                    data.calibration_keys[key], np.eye(3), np.zeros(3), gtsam.noiseModel.Unit.Create(3)
                )
            )
            metadata[key] = replace(info, calibration=info.calibration.retract(np.array([change, 0.002 * change, 0])))
        leaves.append(
            Tree((Path(str(index)), replace(data, camera_graph=graph, camera_metadata=metadata, gauge_fixed=True)))
        )
    tree = Tree((Path("root"), None), tuple(leaves))
    result = cto.optimize_camera_factor_tree(tree, FIXED_OPTIONS)  # Consumer follows the input layout.
    assert result.final_error < result.initial_error
    for i in range(4):
        actual = result.scene.get_camera(i).calibration()
        expected = scene.get_camera(i).calibration().retract(np.array([2, 0.004, 0]))
        np.testing.assert_allclose(actual.vector(), expected.vector(), atol=1e-6)
        assert (actual.px(), actual.py()) == (320, 240)
    graph = gtsam.GaussianFactorGraph(result.camera_bayes_tree)
    assert set(graph.keyVector()) == {key for i in range(4) for key in (X(i), K(i))}
    assert graph.jacobian()[0].shape[1] == 36
    for leaf in result.factor_tree.value.leaves:
        assert set(leaf.estimated_calibrations.keys()) == {K(i) for i in range(4)}


@pytest.fixture
def joint_camera_factors() -> cto.CameraLeafFactors:
    """Produce an observable pose/intrinsics quadratic for contract tests."""
    return cto.compute_camera_schur_complement(_self_calibration_scene())


@pytest.mark.parametrize("case", ["partial", "duplicate", "collision", "missing", "dimension"])
def test_invalid_calibration_mapping(joint_camera_factors: cto.CameraLeafFactors, case: str) -> None:
    """Reject ambiguous key layouts and calibration blocks with the wrong dimension."""
    data = joint_camera_factors
    mapping = dict(data.calibration_keys)
    if case == "partial":
        del mapping[0]
    elif case == "duplicate":
        mapping[0] = mapping[1]
    elif case == "collision":
        mapping[0] = 0
    elif case == "missing":
        mapping[0] = 123
    else:
        graph = data.camera_graph.clone()
        graph.push_back(gtsam.JacobianFactor(K(0), np.eye(6), np.zeros(6), gtsam.noiseModel.Unit.Create(6)))
        data = replace(data, camera_graph=graph)
    with pytest.raises(ValueError, match="Leaf invalid"):
        cto.optimize_camera_factor_tree(Tree((Path("invalid"), replace(data, calibration_keys=mapping))))


def test_inconsistent_principal_points(joint_camera_factors: cto.CameraLeafFactors) -> None:
    """Principal points cannot vary across summaries of the same image."""
    metadata = dict(joint_camera_factors.camera_metadata)
    metadata[0] = replace(metadata[0], calibration=gtsam.Cal3Bundler(400, 0.01, -0.005, 321, 240))
    tree = Tree(
        (Path("root"), None),
        (
            Tree((Path("a"), joint_camera_factors)),
            Tree((Path("b"), replace(joint_camera_factors, camera_metadata=metadata))),
        ),
    )
    with pytest.raises(ValueError, match="Inconsistent principal point"):
        cto.optimize_camera_factor_tree(tree)


def test_mixed_intrinsics_modes(joint_camera_factors: cto.CameraLeafFactors) -> None:
    """A consumer accepts legacy trees but rejects ambiguous mixed variable layouts."""
    fixed = cto.compute_camera_schur_complement(_self_calibration_scene(), FIXED_OPTIONS)
    tree = Tree((Path("root"), None), (Tree((Path("a"), fixed)), Tree((Path("b"), joint_camera_factors))))
    with pytest.raises(ValueError, match="Cannot mix"):
        cto.optimize_camera_factor_tree(tree)


def test_unobservable_intrinsics_are_not_regularized() -> None:
    """Parallel undistorted views have extra calibration freedoms despite observable fixed poses."""
    scene = _synthetic_scene()
    cto.compute_camera_schur_complement(scene, FIXED_OPTIONS)
    with pytest.raises(ValueError, match="Insufficient geometric rank"):
        cto.compute_camera_schur_complement(scene)
