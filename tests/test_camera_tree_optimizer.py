"""Numerical and real-reconstruction tests for camera-only hierarchical fusion."""

from pathlib import Path
from unittest.mock import patch

import gtsam
import numpy as np
import pytest
from gtsam.symbol_shorthand import L, S, X

from gtsfm import camera_tree_optimizer as cto
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.products import scene_tree
from gtsfm.utils.tree import PreOrderIter, Tree

DATA_ROOT = Path(__file__).resolve().parent / "data"


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
    return cto._leaf_inputs(Tree((Path("leaf"), scene)))[0]


def _vector_values(keys: list[int], delta: np.ndarray) -> gtsam.VectorValues:
    """Pack six-dimensional camera increments in a GTSAM VectorValues."""
    values = gtsam.VectorValues()
    for i, key in enumerate(keys):
        values.insert(key, delta[6 * i : 6 * (i + 1)])
    return values


def test_landmark_elimination_and_bayes_tree_preserve_quadratic() -> None:
    """Check the Schur objective against explicit minimization of all point increments."""
    leaf = _single_leaf(_synthetic_scene())
    options = cto.CameraTreeOptions()
    graph, references = cto._linearize_leaf(leaf, options)
    point_ordering = gtsam.Ordering([L(j) for j in range(leaf.scene.number_tracks())])
    camera_keys = list(references.keys())
    full_ordering = gtsam.Ordering()
    for key in camera_keys:
        full_ordering.push_back(key)
    for j in range(leaf.scene.number_tracks()):
        full_ordering.push_back(L(j))
    matrix, rhs = graph.jacobian(full_ordering)
    camera_matrix = matrix[:, : 6 * len(camera_keys)]
    point_matrix = matrix[:, 6 * len(camera_keys) :]
    _, reduced = graph.eliminatePartialMultifrontal(point_ordering, gtsam.EliminateQR)
    assert set(reduced.keyVector()) == set(camera_keys)
    summary = cto._condense_leaf(leaf, options)
    tree_graph = gtsam.GaussianFactorGraph(summary.camera_bayes_tree)
    zero = _vector_values(camera_keys, np.zeros(6 * len(camera_keys)))
    zero_points = np.linalg.lstsq(point_matrix, rhs, rcond=None)[0]
    zero_profile = 0.5 * np.linalg.norm(point_matrix @ zero_points - rhs) ** 2
    rng = np.random.default_rng(3)
    anchor = references.atPose3(camera_keys[0])
    farthest = max(
        range(1, len(camera_keys)),
        key=lambda i: np.linalg.norm(references.atPose3(camera_keys[i]).translation() - anchor.translation()),
    )
    pose = references.atPose3(camera_keys[farthest])
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
        np.testing.assert_allclose(
            reduced.error(vector) - reduced.error(zero), expected_difference, rtol=1e-9, atol=1e-8
        )
        np.testing.assert_allclose(
            tree_graph.error(vector) - tree_graph.error(zero), expected_difference, rtol=1e-9, atol=1e-8
        )


def test_similarity_factor_analytic_jacobians() -> None:
    """Check pose and similarity derivatives away from identity and at nonunit scale."""
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(9, 12))
    rhs = rng.normal(size=9)
    gaussian = gtsam.JacobianFactor(X(0), matrix[:, :6], X(1), matrix[:, 6:], rhs, gtsam.noiseModel.Unit.Create(9))
    references = gtsam.Values()
    references.insert(X(0), gtsam.Pose3.Expmap(rng.normal(0, 0.2, 6)))
    references.insert(X(1), gtsam.Pose3.Expmap(rng.normal(0, 0.2, 6)))
    factor = cto._lift_conditional(gaussian, references, S(0))
    values = gtsam.Values()
    similarity = gtsam.Similarity3(gtsam.Rot3.RzRyRx(0.2, -0.3, 0.1), np.array([1.0, -2.0, 0.4]), 2.3)
    values.insert(S(0), similarity)
    for key in references.keys():
        pose = references.atPose3(key).retract(rng.normal(0, 0.1, 6))
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


@pytest.fixture(scope="module")
def lund_result(lund_tree: scene_tree.SceneTree) -> cto.CameraTreeResult:
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
        return cto.optimize_camera_tree(lund_tree)


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
        assert set(keys) == {X(i) for i in leaf.local_to_global.values()}
    assert set(gtsam.GaussianFactorGraph(result.camera_bayes_tree).keyVector()) == {X(i) for i in range(12)}
    for i in range(12):
        assert result.scene.get_image_info(i).name == f"DSC_{i + 1:04}.JPG"
        assert np.all(np.isfinite(result.scene.get_camera(i).pose().matrix()))


def test_lund_independent_leaf_frame_invariance(
    lund_tree: scene_tree.SceneTree, lund_result: cto.CameraTreeResult
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
    result = cto.optimize_camera_tree(reframed)
    reference_frame = transforms[0].inverse()
    for i in range(12):
        expected = lund_result.scene.get_camera(i).pose()
        actual = reference_frame.transformFrom(result.scene.get_camera(i).pose())
        np.testing.assert_allclose(expected.localCoordinates(actual), np.zeros(6), atol=2e-5)
    np.testing.assert_allclose(result.final_error, lund_result.final_error, rtol=1e-6, atol=1e-6)


def test_perturbed_initialization_and_ignored_internal_scenes(lund_tree: scene_tree.SceneTree) -> None:
    """The root solve must correct perturbations without accessing internal scene payloads."""
    initialize = cto._initialize

    def perturb(leaves: tuple[cto._LeafInput, ...]) -> gtsam.Values:
        """Perturb camera estimates but keep the reference similarity gauge feasible."""
        values = initialize(leaves)
        rng = np.random.default_rng(2)
        for key in values.keys():
            if gtsam.Symbol(key).chr() == ord("x"):
                values.update(key, values.atPose3(key).retract(rng.normal(0, 1e-4, 6)))
            elif key != S(0):
                values.update(key, values.atSimilarity3(key).retract(rng.normal(0, 1e-4, 7)))
        return values

    poisoned = lund_tree.map(lambda value: (value[0], value[1] if value[1] is not None else object()))
    with patch.object(cto, "_initialize", side_effect=perturb):
        result = cto.optimize_camera_tree(poisoned)
    assert result.final_error < result.initial_error
    assert result.scene.number_images() == 12


def test_original_lund_leaves_are_unobservable() -> None:
    """The unchanged fixture has four disjoint camera sets and cannot determine relative scales."""
    tree = scene_tree.read_colmap(str(DATA_ROOT / "lund_door_binary"))
    assert tree is not None
    with pytest.raises(ValueError, match="Disconnected or insufficient camera overlap.*C_1/C_1_2"):
        cto.optimize_camera_tree(tree)


def test_one_shared_camera_is_insufficient() -> None:
    """A single shared camera does not fix the relative leaf scale."""
    tree = Tree(
        (Path("root"), None),
        (Tree((Path("a"), _synthetic_scene((0, 1, 2)))), Tree((Path("b"), _synthetic_scene((2, 3, 4))))),
    )
    with pytest.raises(ValueError, match="two shared cameras"):
        cto.optimize_camera_tree(tree)


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


@pytest.mark.parametrize("kwargs", [{"measurement_sigma": 0}, {"max_iterations": 0}, {"rank_tolerance": float("nan")}])
def test_invalid_options(kwargs: dict) -> None:
    """Reject invalid numerical options at construction."""
    with pytest.raises(ValueError):
        cto.CameraTreeOptions(**kwargs)
