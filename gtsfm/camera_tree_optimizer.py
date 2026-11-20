"""Fuse frozen leaf BA information in a common frame without retaining landmarks.

Each leaf contributes a camera-only Gaussian Bayes tree in its own coordinate
gauge. Similarity-aware factors carry its full joint camera information into a
single root optimization. The result approximates BA at the supplied leaf
linearizations; no reprojection factors are evaluated after condensation.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, S, X
from numpy.typing import NDArray

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.products.scene_tree import LocalScene, SceneTree
from gtsfm.utils import align
from gtsfm.utils.tree import Tree


@dataclass(frozen=True)
class CameraTreeOptions:
    """Fusion settings; measurement_sigma is used only by the Schur-complement producer."""

    measurement_sigma: float = 1.0
    max_iterations: int = 100
    relative_error_tol: float = 1e-9
    absolute_error_tol: float = 1e-9
    rank_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Reject invalid noise, iteration, and numerical tolerance settings."""
        if not np.isfinite(self.measurement_sigma) or self.measurement_sigma <= 0:
            raise ValueError("measurement_sigma must be finite and positive.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        for name in ("relative_error_tol", "absolute_error_tol", "rank_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and between zero and one.")


@dataclass(frozen=True)
class LeafCameraMetadata:
    """Camera identity and fixed calibration, independent of any reconstruction."""

    filename: str
    calibration: gtsam.Cal3Bundler
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class CameraLeafFactors:
    """A complete camera quadratic in Pose3 local coordinates at fixed leaf estimates.

    All fields use the same leaf-local keys. The graph must retain its linear term
    and measurement weighting. Set gauge_fixed only when the producer has already
    fixed the seven coordinate freedoms without adding unintended geometric priors.
    """

    camera_graph: gtsam.GaussianFactorGraph
    estimated_camera_poses: gtsam.Values
    camera_metadata: dict[int, LeafCameraMetadata]
    gauge_fixed: bool = False


LocalCameraFactors = tuple[Path, CameraLeafFactors | None]
CameraFactorTree = Tree[LocalCameraFactors]


@dataclass(frozen=True)
class LeafCameraSummary:
    """Landmark-free leaf model; estimated camera poses stay fixed and are keyed by global X(i)."""

    path: Path
    similarity_key: int
    local_to_global: dict[int, int]
    estimated_camera_poses: gtsam.Values
    camera_bayes_tree: gtsam.GaussianBayesTree
    factors: tuple[gtsam.CustomFactor, ...]


@dataclass(frozen=True)
class CameraSubtreeSummary:
    """Factors and leaf summaries accumulated once from a subtree."""

    graph: gtsam.NonlinearFactorGraph
    leaves: tuple[LeafCameraSummary, ...]


@dataclass(frozen=True)
class CameraTreeResult:
    """Optimized cameras and linearized uncertainty in the first leaf's frame.

    ``scene`` has no tracks. ``leaf_to_global`` maps each input leaf frame into
    the output frame. The final Bayes tree includes the chosen global gauge.
    Calibration and image metadata come from the first occurrence of a camera.
    """

    scene: GtsfmData
    leaf_to_global: dict[Path, gtsam.Similarity3]
    factor_tree: Tree[CameraSubtreeSummary]
    camera_bayes_tree: gtsam.GaussianBayesTree
    initial_error: float
    final_error: float
    iterations: int


@dataclass(frozen=True)
class _LeafInput:
    """Validated leaf input with image identities resolved across the tree."""

    path: Path
    data: CameraLeafFactors
    local_to_global: dict[int, int]
    similarity_key: int


def _validate_camera_metadata(
    estimated_camera_poses: gtsam.Values, camera_metadata: dict[int, LeafCameraMetadata]
) -> None:
    """Validate camera identities and estimates shared by both public input routes.

    Args:
        estimated_camera_poses: Fixed leaf-local Pose3 estimates.
        camera_metadata: Identity and calibration for each camera key.
    """
    if set(estimated_camera_poses.keys()) != set(camera_metadata):
        raise ValueError("Camera pose keys and metadata keys must match.")
    if estimated_camera_poses.size() < 2:
        raise ValueError("Need at least two reconstructed cameras.")
    names: set[str] = set()
    for key, info in camera_metadata.items():
        if not isinstance(info.filename, str) or not info.filename or info.filename in names:
            raise ValueError(f"missing or duplicate camera filename at key {key}.")
        names.add(info.filename)
        if not isinstance(info.calibration, gtsam.Cal3Bundler):
            raise ValueError("Only Cal3Bundler cameras are supported.")
        if not np.all(np.isfinite(info.calibration.K())) or not np.all(np.isfinite(info.calibration.vector())):
            raise ValueError(f"Nonfinite calibration at camera key {key}.")
        try:
            pose = estimated_camera_poses.atPose3(key)
        except RuntimeError as exc:
            raise ValueError(f"Camera key {key} must contain a Pose3 estimate.") from exc
        if not np.all(np.isfinite(pose.matrix())):
            raise ValueError(f"Nonfinite camera pose at key {key}.")


def _leaf_inputs(camera_factor_tree: CameraFactorTree) -> tuple[_LeafInput, ...]:
    """Resolve camera identity by filename, inspecting only terminal nodes.

    Args:
        camera_factor_tree: Tree of optional camera quadratics.

    Returns:
        Validated leaf inputs in deterministic tree traversal order.
    """
    names: set[str] = set()
    paths: set[Path] = set()
    for node in camera_factor_tree.leaves():
        path, data = node.value
        if path in paths:
            raise ValueError(f"Duplicate leaf path: {path}.")
        paths.add(path)
        if data is None:
            raise ValueError(f"Leaf {path}: missing camera factors.")
        try:
            _validate_camera_metadata(data.estimated_camera_poses, data.camera_metadata)
            if set(data.camera_graph.keyVector()) != set(data.camera_metadata):
                raise ValueError("Camera graph keys and metadata keys must match.")
            if not isinstance(data.gauge_fixed, bool):
                raise ValueError("gauge_fixed must be a bool.")
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Leaf {path}: {exc}") from exc
        names.update(info.filename for info in data.camera_metadata.values())
    global_indices = {name: i for i, name in enumerate(sorted(names))}
    inputs = []
    for leaf_index, node in enumerate(camera_factor_tree.leaves()):
        path, data = node.value
        assert data is not None
        mapping = {key: global_indices[info.filename] for key, info in data.camera_metadata.items()}
        inputs.append(_LeafInput(path, data, mapping, S(leaf_index)))
    return tuple(inputs)


def _local_poses(leaf: _LeafInput) -> dict[int, gtsam.Pose3]:
    """Get local poses indexed by their global camera identities.

    Args:
        leaf: Camera factors with resolved camera identities.

    Returns:
        Global camera indices mapped to poses in the leaf frame.
    """
    return {global_i: leaf.data.estimated_camera_poses.atPose3(key) for key, global_i in leaf.local_to_global.items()}


def _has_baseline(poses: dict[int, gtsam.Pose3], indices: list[int]) -> bool:
    """Check that a group of cameras includes two distinct centers.

    Args:
        poses: Camera poses indexed by global camera identity.
        indices: Cameras to consider for the baseline.

    Returns:
        Whether at least two selected camera centers are distinct.
    """
    if len(indices) < 2:
        return False
    centers = np.array([poses[i].translation() for i in indices])
    return bool(np.max(np.linalg.norm(centers - centers[0], axis=1)) > 1e-10)


def _initialize(leaves: tuple[_LeafInput, ...]) -> gtsam.Values:
    """Initialize a connected overlap graph by pose-based similarity alignment.

    Args:
        leaves: Validated camera factors with global camera identities.

    Returns:
        Global Pose3 values and similarities mapping global into leaf frames.
    """
    poses = [_local_poses(leaf) for leaf in leaves]
    for leaf, local in zip(leaves, poses):
        if not _has_baseline(local, sorted(local)):
            raise ValueError(f"Leaf {leaf.path}: degenerate camera baseline.")
    values = gtsam.Values()
    values.insert(leaves[0].similarity_key, gtsam.Similarity3())
    for i, pose in poses[0].items():
        values.insert(X(i), pose)
    visited = [0]
    pending = set(range(1, len(leaves)))
    for parent_index in visited:
        parent_to_global = values.atSimilarity3(leaves[parent_index].similarity_key).inverse()
        parent_poses = {i: parent_to_global.transformFrom(pose) for i, pose in poses[parent_index].items()}
        for child_index in sorted(pending):
            child_poses = poses[child_index]
            common = sorted(parent_poses.keys() & child_poses.keys())
            if not _has_baseline(parent_poses, common) or not _has_baseline(child_poses, common):
                continue
            child_to_global = align.sim3_from_Pose3_maps(parent_poses, child_poses)
            if not np.isfinite(child_to_global.scale()) or child_to_global.scale() <= 0:
                raise ValueError(f"Leaf {leaves[child_index].path}: invalid initial similarity scale.")
            values.insert(leaves[child_index].similarity_key, child_to_global.inverse())
            for i, pose in child_poses.items():
                if not values.exists(X(i)):
                    values.insert(X(i), child_to_global.transformFrom(pose))
            pending.remove(child_index)
            visited.append(child_index)
    if pending:
        paths = ", ".join(str(leaves[i].path) for i in sorted(pending))
        raise ValueError(
            f"Disconnected or insufficient camera overlap for leaves: {paths}. "
            "Each overlap link needs two shared cameras with distinct centers."
        )
    return values


def _reconstruction_camera_data(reconstruction: GtsfmData) -> tuple[gtsam.Values, dict[int, LeafCameraMetadata]]:
    """Extract camera-only estimates and metadata from a reconstruction.

    Args:
        reconstruction: Reconstructed cameras and tracks.

    Returns:
        Pose estimates and metadata keyed by reconstruction camera indices.
    """
    estimated_camera_poses = gtsam.Values()
    metadata = {}
    for i in reconstruction.get_valid_camera_indices():
        camera = reconstruction.get_camera(i)
        estimated_camera_poses.insert(i, camera.pose())
        info = reconstruction.get_image_info(i)
        metadata[i] = LeafCameraMetadata(info.name, camera.calibration(), info.shape)
    _validate_camera_metadata(estimated_camera_poses, metadata)
    poses = {key: estimated_camera_poses.atPose3(key) for key in estimated_camera_poses.keys()}
    if not _has_baseline(poses, sorted(poses)):
        raise ValueError("degenerate camera baseline.")
    return estimated_camera_poses, metadata


def _fixed_camera_values(reconstruction: GtsfmData) -> gtsam.Values:
    """Create six-dimensional fixed-calibration camera values for reprojection factors.

    Args:
        reconstruction: Input with validated Cal3Bundler cameras.

    Returns:
        PinholePose values keyed by reconstruction camera indices.
    """
    values = gtsam.Values()
    for i in reconstruction.get_valid_camera_indices():
        camera = reconstruction.get_camera(i)
        calibration = camera.calibration()
        # The PinholePose constructor requires an owned/shared calibration object.
        held_calibration = gtsam.Cal3Bundler(
            calibration.fx(), calibration.k1(), calibration.k2(), calibration.px(), calibration.py()
        )
        values.insert(i, gtsam.PinholePoseCal3Bundler(camera.pose(), held_calibration))
    return values


def _linearize_reconstruction(reconstruction: GtsfmData, options: CameraTreeOptions) -> gtsam.GaussianFactorGraph:
    """Linearize fixed-calibration observations once without running bundle adjustment.

    Args:
        reconstruction: Input with validated cameras and landmark estimates.
        options: Measurement noise settings.

    Returns:
        Joint camera/landmark Gaussian graph in the input frame.
    """
    values = _fixed_camera_values(reconstruction)
    camera_keys = set(values.keys())
    graph = gtsam.NonlinearFactorGraph()
    noise = gtsam.noiseModel.Isotropic.Sigma(2, options.measurement_sigma)
    for j, track in enumerate(reconstruction.tracks()):
        measurements = [track.measurement(k) for k in range(track.numberMeasurements())]
        measurements = [(i, uv) for i, uv in measurements if i in camera_keys]
        if len({i for i, _ in measurements}) < 2:
            continue
        if not np.all(np.isfinite(track.point3())) or not all(np.all(np.isfinite(uv)) for _, uv in measurements):
            raise ValueError(f"Nonfinite point or measurement in track {j}.")
        values.insert(L(j), track.point3())
        for i, uv in measurements:
            graph.push_back(gtsam.GeneralSFMFactorPoseCal3Bundler(uv, noise, i, L(j)))
    if graph.empty():
        raise ValueError("No tracks with at least two camera observations.")
    return graph.linearize(values)


def _camera_marginal_keys(
    graph: gtsam.NonlinearFactorGraph, values: gtsam.Values, camera_keys: Sequence[int] | None
) -> list[int]:
    """Validate the original graph and identify the camera Pose3 variables to retain.

    Args:
        graph: Original BA graph, including its gauge factors.
        values: Estimates at which to linearize the original factors.
        camera_keys: Explicit keys, or None to select conventional X(i) keys.

    Returns:
        Distinct camera keys, sorted for deterministic input handling.
    """
    graph_keys = set(graph.keyVector())
    if not graph_keys.issubset(values.keys()):
        raise ValueError("Values must contain every variable in the original factor graph.")
    keys = (
        list(camera_keys) if camera_keys is not None else [k for k in graph_keys if gtsam.Symbol(k).chr() == ord("x")]
    )
    if not keys or len(set(keys)) != len(keys) or not set(keys).issubset(graph_keys):
        raise ValueError("Camera keys must be nonempty, distinct, and present in the original factor graph.")
    for key in keys:
        try:
            pose = values.atPose3(key)
        except RuntimeError as exc:
            raise ValueError(f"Camera key {key} must contain a six-dimensional Pose3.") from exc
        if not np.all(np.isfinite(pose.matrix())):
            raise ValueError(f"Nonfinite camera pose at key {key}.")
    return sorted(keys)


def _eliminate_non_camera_variables(
    linear_graph: gtsam.GaussianFactorGraph, camera_keys: list[int]
) -> gtsam.GaussianFactorGraph:
    """Eliminate landmarks and other nuisance variables before any camera.

    Args:
        linear_graph: Original graph linearized at the supplied estimates.
        camera_keys: Pose keys to keep after Schur elimination.

    Returns:
        Complete reduced camera quadratic, before camera factorization.
    """
    ordering = gtsam.Ordering.ColamdConstrainedLastGaussianFactorGraph(linear_graph, camera_keys)
    nuisance_count = ordering.size() - len(camera_keys)
    nuisance_ordering = gtsam.Ordering([ordering.at(k) for k in range(nuisance_count)])
    _, camera_graph = linear_graph.eliminatePartialMultifrontal(nuisance_ordering, gtsam.EliminateQR)
    return camera_graph


def compute_camera_marginal_factors(
    graph: gtsam.NonlinearFactorGraph, values: gtsam.Values, camera_keys: Sequence[int] | None = None
) -> gtsam.GaussianFactorGraph:
    """Return compact camera marginal factors from the original BA graph and estimates.

    The supplied graph must include its coordinate gauge. All non-camera variables
    are eliminated first; camera conditionals follow METIS nested dissection.
    Factors retain the linear term, measurement weights, and exact constraints.
    No BA or additional gauge fixing is performed, and inputs are not modified.

    Args:
        graph: Original nonlinear graph, including its gauge and other priors.
        values: Estimated poses, landmarks, and any other variables in the graph.
        camera_keys: Pose3 keys to retain; defaults to the graph's X(i) keys.

    Returns:
        One Gaussian conditional per camera clique, using the supplied camera keys and
        Pose3 local coordinates at values. Constants discarded during elimination
        are excluded. Sparse camera connectivity is preserved up to elimination fill.
        Use gauge_fixed=True when packaging this graph in CameraLeafFactors.

    Raises:
        ValueError: Keys, estimates, factors, or gauge are invalid, or GTSAM lacks METIS support.
    """
    keys = _camera_marginal_keys(graph, values, camera_keys)
    try:
        # Linearize the original factors once, preserving their weights and gauge.
        camera_graph = _eliminate_non_camera_variables(graph.linearize(values), keys)
        # Compress the Schur complement in nested-dissection order using multifrontal QR.
        camera_ordering = gtsam.Ordering.MetisGaussianFactorGraph(camera_graph)
        compact = gtsam.GaussianFactorGraph(camera_graph.eliminateMultifrontal(camera_ordering, gtsam.EliminateQR))
        _validate_camera_system(compact, len(keys), True, CameraTreeOptions())
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Camera marginalization failed (the graph must include a full coordinate gauge): {exc}"
        ) from exc
    return compact


def compute_camera_schur_complement(
    reconstruction: GtsfmData, options: CameraTreeOptions | None = None
) -> CameraLeafFactors:
    """Produce an ungauged camera quadratic by linearizing once and eliminating landmarks.

    Warning:
        This convenience API rebuilds observations from GtsfmData, which does not
        retain the original BA graph, gauge, or noise models. Prefer
        compute_camera_marginal_factors() with the original graph and estimates.

    Args:
        reconstruction: Named reconstructed Cal3Bundler cameras and tracks.
        options: Numerical settings; measurement_sigma defaults to one pixel.

    Returns:
        Camera-only factors, estimated poses, and metadata keyed by input camera indices.
        The complete linear term and weighting are retained; gauge_fixed is False.

    Raises:
        ValueError: Camera identities, geometry, or observations are invalid.
    """
    options = options or CameraTreeOptions()
    estimated_camera_poses, metadata = _reconstruction_camera_data(reconstruction)
    try:
        linear_graph = _linearize_reconstruction(reconstruction, options)
        if not set(metadata).issubset(linear_graph.keyVector()):
            raise ValueError("Some cameras have no retained observations.")
        camera_graph = linear_graph.marginal(list(estimated_camera_poses.keys()), gtsam.EliminateQR)
        data = CameraLeafFactors(camera_graph, estimated_camera_poses, metadata)
        _validate_camera_graph(data, options)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Camera Schur complement failed: {exc}") from exc
    return data


def _add_coordinate_gauge(graph: gtsam.GaussianFactorGraph, estimated_camera_poses: gtsam.Values) -> None:
    """Fix one local pose and one baseline direction using exact coordinate constraints.

    Args:
        graph: Camera graph to augment with seven constrained rows.
        estimated_camera_poses: Local estimated camera poses, keyed by global camera keys.
    """
    keys = list(estimated_camera_poses.keys())
    anchor_key = keys[0]
    anchor = estimated_camera_poses.atPose3(anchor_key)
    baseline_key = max(
        keys[1:],
        key=lambda key: np.linalg.norm(estimated_camera_poses.atPose3(key).translation() - anchor.translation()),
    )
    baseline_pose = estimated_camera_poses.atPose3(baseline_key)
    baseline_length = np.linalg.norm(baseline_pose.translation() - anchor.translation())
    pose_constraint = gtsam.NonlinearEqualityPose3(anchor_key, anchor)
    baseline_constraint = gtsam.RangeFactorPose3(
        anchor_key, baseline_key, baseline_length, gtsam.noiseModel.Constrained.All(1)
    )
    # Freeze the standard factors into the same local coordinates as the leaf camera quadratic.
    graph.push_back(pose_constraint.linearize(estimated_camera_poses))
    graph.push_back(baseline_constraint.linearize(estimated_camera_poses))


def _lift_conditional(
    conditional: gtsam.GaussianFactor, estimated_camera_poses: gtsam.Values, similarity_key: int
) -> gtsam.CustomFactor:
    """Evaluate a camera conditional through a global-to-leaf similarity.

    Args:
        conditional: Gaussian camera conditional with six columns per camera.
        estimated_camera_poses: Frozen local estimated camera poses.
        similarity_key: Key for the global-to-leaf Similarity3 variable.

    Returns:
        Nonlinear factor preserving the conditional's original noise model.
    """
    factor = gtsam.JacobianFactor(conditional)
    keys = factor.keys()
    matrix = np.array(factor.getA())
    rhs = np.array(factor.getb())
    ordered_estimated_camera_poses = tuple(estimated_camera_poses.atPose3(key) for key in keys)

    def error(
        this: gtsam.CustomFactor, values: gtsam.Values, jacobians: gtsam.JacobianVector | None
    ) -> NDArray[np.float64]:
        """Evaluate residuals and analytic right-retraction Jacobians.

        Args:
            this: Factor supplied by GTSAM's callback interface.
            values: Current global poses and global-to-leaf similarity.
            jacobians: Optional output blocks, ordered as similarity then cameras.

        Returns:
            Unwhitened conditional residual vector.
        """
        similarity = values.atSimilarity3(similarity_key)
        deltas = []
        if jacobians is not None:
            jacobians[0] = np.zeros((matrix.shape[0], 7), order="F")
        for column, (key, estimated_camera_pose) in enumerate(zip(keys, ordered_estimated_camera_poses)):
            pose = values.atPose3(key)
            if jacobians is None:
                deltas.append(estimated_camera_pose.localCoordinates(similarity.transformFrom(pose)))
                continue
            h_similarity = np.zeros((6, 7), order="F")
            h_pose = np.zeros((6, 6), order="F")
            h_local = np.zeros((6, 6), order="F")
            local_pose = similarity.transformFrom(pose, h_similarity, h_pose)
            deltas.append(estimated_camera_pose.localCoordinates(local_pose, None, h_local))
            block = matrix[:, 6 * column : 6 * (column + 1)] @ h_local
            jacobians[0] = jacobians[0] + block @ h_similarity
            jacobians[column + 1] = block @ h_pose
        return matrix @ np.concatenate(deltas) - rhs

    return gtsam.CustomFactor(factor.get_model(), [similarity_key, *keys], error)


def _validate_camera_factor(factor: gtsam.GaussianFactor) -> None:
    """Check individual block dimensions and numerical data before combining factors.

    Args:
        factor: A Gaussian camera factor, possibly containing exact constraints.
    """
    # A singleton gradient reveals each block's dimension without assuming equal block sizes.
    singleton = gtsam.GaussianFactorGraph()
    singleton.push_back(factor)
    dimensions = singleton.gradientAtZero()
    if any(dimensions.at(key).size != 6 for key in factor.keys()):
        raise ValueError("Every camera factor block must have six dimensions.")
    jacobian = gtsam.JacobianFactor(factor)
    if not np.all(np.isfinite(jacobian.getA())) or not np.all(np.isfinite(jacobian.getb())):
        raise ValueError("Nonfinite camera factor coefficients or residual offsets.")
    noise = jacobian.get_model()
    if noise is not None and (not np.all(np.isfinite(noise.sigmas())) or np.any(noise.sigmas() < 0)):
        raise ValueError("Camera factor noise sigmas must be finite and nonnegative.")


def _validate_camera_graph(data: CameraLeafFactors, options: CameraTreeOptions) -> None:
    """Check factor dimensions and the declared number of coordinate freedoms.

    Args:
        data: Complete camera quadratic and gauge declaration.
        options: Numerical rank tolerance.
    """
    _validate_camera_system(data.camera_graph, data.estimated_camera_poses.size(), data.gauge_fixed, options)


def _validate_camera_system(
    camera_graph: gtsam.GaussianFactorGraph, number_cameras: int, gauge_fixed: bool, options: CameraTreeOptions
) -> None:
    """Check camera blocks and rank independently of reconstruction metadata.

    Args:
        camera_graph: Reduced camera factors, including residual offsets.
        number_cameras: Number of six-dimensional pose variables.
        gauge_fixed: Whether coordinate constraints are already present.
        options: Numerical rank tolerance.
    """
    for k in range(camera_graph.size()):
        factor = camera_graph.at(k)
        if factor is not None:
            _validate_camera_factor(factor)
    matrix, _ = camera_graph.jacobian()
    if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("Camera graph must have finite, nonempty Jacobian rows.")
    column_norms = np.linalg.norm(matrix, axis=0)
    scaled = matrix / np.where(column_norms > 0, column_norms, 1)
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    rank = np.count_nonzero(singular_values > options.rank_tolerance * singular_values[0])
    expected_rank = 6 * number_cameras - (0 if gauge_fixed else 7)
    # Rotation rounding and Hessian square roots can lift null modes by sqrt(n * epsilon).
    gauge_tolerance = max(options.rank_tolerance, np.sqrt(matrix.shape[1] * np.finfo(float).eps))
    gauge_rank = np.count_nonzero(singular_values > gauge_tolerance * singular_values[0])
    if rank < expected_rank or (gauge_fixed and gauge_rank < expected_rank) or gauge_rank > expected_rank:
        raise ValueError(
            f"Insufficient geometric rank or incorrect gauge_fixed={gauge_fixed}: "
            f"rank {rank}, expected {expected_rank}."
        )


def _global_camera_factors(leaf: _LeafInput) -> tuple[gtsam.GaussianFactorGraph, gtsam.Values]:
    """Copy a leaf's factors and estimates into the shared global camera key space.

    Args:
        leaf: Validated camera-only input with resolved image identities.

    Returns:
        Rekeyed factor graph and leaf-frame poses, without mutating the input.
    """
    graph = gtsam.GaussianFactorGraph()
    for k in range(leaf.data.camera_graph.size()):
        factor = leaf.data.camera_graph.at(k)
        if factor is None:
            continue
        new_keys = [X(leaf.local_to_global[key]) for key in factor.keys()]
        container = gtsam.LinearContainerFactor(factor.clone()).rekey(new_keys)
        graph.push_back(container.factor())
    estimated_camera_poses = gtsam.Values()
    for key, global_i in leaf.local_to_global.items():
        estimated_camera_poses.insert(X(global_i), leaf.data.estimated_camera_poses.atPose3(key))
    return graph, estimated_camera_poses


def _prepare_leaf(leaf: _LeafInput, options: CameraTreeOptions) -> LeafCameraSummary:
    """Factor an externally condensed camera graph and lift its conditionals.

    Args:
        leaf: Camera-only input with resolved image identities.
        options: Geometric rank settings; observation noise is already in the factors.

    Returns:
        A summary with globally keyed camera information and no landmark variables.
    """
    try:
        _validate_camera_graph(leaf.data, options)
        camera_graph, estimated_camera_poses = _global_camera_factors(leaf)
        if not leaf.data.gauge_fixed:
            _add_coordinate_gauge(camera_graph, estimated_camera_poses)
        bayes_tree = camera_graph.eliminateMultifrontal(None, gtsam.EliminateQR)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Leaf {leaf.path}: camera preparation failed: {exc}") from exc
    conditionals = gtsam.GaussianFactorGraph(bayes_tree)
    factors = tuple(
        _lift_conditional(conditionals.at(k), estimated_camera_poses, leaf.similarity_key)
        for k in range(conditionals.size())
    )
    return LeafCameraSummary(
        leaf.path, leaf.similarity_key, leaf.local_to_global, estimated_camera_poses, bayes_tree, factors
    )


def _assemble_subtree(
    value: LocalCameraFactors,
    children: tuple[CameraSubtreeSummary, ...],
    *,
    inputs_by_path: dict[Path, _LeafInput],
    options: CameraTreeOptions,
) -> CameraSubtreeSummary:
    """Combine child graphs once, or prepare a terminal camera quadratic.

    Args:
        value: Node path and optional camera factors.
        children: Summaries already assembled for the immediate children.
        inputs_by_path: Validated inputs for terminal nodes.
        options: Camera factor preparation settings.

    Returns:
        All factors and leaf summaries belonging to this subtree.
    """
    graph = gtsam.NonlinearFactorGraph()
    if children:
        # Internal nodes collect each child's evidence once, ignoring their own payload.
        for child in children:
            graph.push_back(child.graph)
        return CameraSubtreeSummary(graph, tuple(leaf for child in children for leaf in child.leaves))
    # Express this leaf's supplied camera evidence through a similarity.
    summary = _prepare_leaf(inputs_by_path[value[0]], options)
    for factor in summary.factors:
        graph.push_back(factor)
    return CameraSubtreeSummary(graph, (summary,))


def _assemble_factor_tree(
    camera_factor_tree: CameraFactorTree, leaves: tuple[_LeafInput, ...], options: CameraTreeOptions
) -> Tree[CameraSubtreeSummary]:
    """Assemble all leaf factors bottom-up while preserving the input hierarchy.

    Args:
        camera_factor_tree: Hierarchy to traverse.
        leaves: Validated camera-only leaf inputs.
        options: Camera factor preparation settings.

    Returns:
        Mirrored tree containing the accumulated camera factors at each node.
    """
    assemble = partial(_assemble_subtree, inputs_by_path={leaf.path: leaf for leaf in leaves}, options=options)
    return camera_factor_tree.map_with_children(assemble)


def _optimize_root(
    graph: gtsam.NonlinearFactorGraph, initial: gtsam.Values, options: CameraTreeOptions
) -> gtsam.LevenbergMarquardtOptimizer:
    """Optimize the joint camera-and-similarity graph once and check convergence.

    Args:
        graph: Assembled factors including the global coordinate constraint.
        initial: Initial global poses and global-to-leaf similarities.
        options: Root optimization settings.

    Returns:
        Completed optimizer containing the solution, objective, and iteration count.
    """
    # QR accommodates the exact coordinate constraints in the leaf factors.
    params = gtsam.LevenbergMarquardtParams()
    params.setLinearSolverType("MULTIFRONTAL_QR")
    params.setMaxIterations(options.max_iterations)
    params.setRelativeErrorTol(options.relative_error_tol)
    params.setAbsoluteErrorTol(options.absolute_error_tol)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    optimized = optimizer.optimize()
    final_error = graph.error(optimized)
    if not np.isfinite(final_error) or optimizer.iterations() >= options.max_iterations:
        raise RuntimeError(
            f"Camera tree optimization did not converge: iterations={optimizer.iterations()}, error={final_error}."
        )
    return optimizer


def _final_camera_bayes_tree(
    graph: gtsam.NonlinearFactorGraph, optimized: gtsam.Values, leaves: tuple[_LeafInput, ...]
) -> gtsam.GaussianBayesTree:
    """Reduce the optimized joint graph to a Gaussian Bayes tree on cameras only.

    Args:
        graph: Assembled factors including the global coordinate constraint.
        optimized: Converged global camera and similarity values.
        leaves: Leaf inputs identifying the similarity keys to eliminate.

    Returns:
        Camera-only Bayes tree in the chosen global gauge.
    """
    # Remove similarity increments from the final linearization, retaining their camera information.
    camera_keys = sorted({X(i) for leaf in leaves for i in leaf.local_to_global.values()})
    final_camera_graph = graph.linearize(optimized).marginal(camera_keys, gtsam.EliminateQR)
    # Factor the remaining camera quadratic with GTSAM's default COLAMD ordering.
    return final_camera_graph.eliminateMultifrontal(None, gtsam.EliminateQR)


def _camera_scene(leaves: tuple[_LeafInput, ...], optimized: gtsam.Values) -> GtsfmData:
    """Build a camera-only scene from optimized poses and source metadata.

    Args:
        leaves: Camera-only inputs providing calibration and image metadata.
        optimized: Converged global camera and similarity values.

    Returns:
        Global cameras with image metadata and no landmarks.
    """
    # Export each global camera once, keeping its first leaf's calibration and image metadata.
    scene = GtsfmData(number_images=sum(gtsam.Symbol(key).chr() == ord("x") for key in optimized.keys()))
    for leaf in leaves:
        for local_i, global_i in leaf.local_to_global.items():
            if scene.get_camera(global_i) is not None:
                continue
            info = leaf.data.camera_metadata[local_i]
            scene.add_camera(global_i, gtsam.PinholeCameraCal3Bundler(optimized.atPose3(X(global_i)), info.calibration))
            scene.set_image_info(global_i, name=info.filename, shape=info.shape)
    return scene


def _camera_tree_result(
    factor_tree: Tree[CameraSubtreeSummary],
    leaves: tuple[_LeafInput, ...],
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    optimizer: gtsam.LevenbergMarquardtOptimizer,
) -> CameraTreeResult:
    """Package global cameras, leaf transforms, and the final camera quadratic.

    Args:
        factor_tree: Assembled hierarchy of leaf camera factors.
        leaves: Validated leaf inputs and camera identity mappings.
        graph: Root graph including the global coordinate constraint.
        initial: Initial values for computing the starting objective.
        optimizer: Completed root optimizer.

    Returns:
        Camera-only reconstruction and optimization summaries.
    """
    optimized = optimizer.values()
    # Invert the internal global-to-leaf similarities for the public leaf-to-global result.
    return CameraTreeResult(
        scene=_camera_scene(leaves, optimized),
        leaf_to_global={leaf.path: optimized.atSimilarity3(leaf.similarity_key).inverse() for leaf in leaves},
        factor_tree=factor_tree,
        camera_bayes_tree=_final_camera_bayes_tree(graph, optimized, leaves),
        initial_error=graph.error(initial),
        final_error=graph.error(optimized),
        iterations=optimizer.iterations(),
    )


def optimize_camera_factor_tree(
    camera_factor_tree: CameraFactorTree, options: CameraTreeOptions | None = None
) -> CameraTreeResult:
    """Fuse frozen leaf camera information in the first leaf's frame without further BA.

    Intrinsics stay fixed, and internal payloads are ignored.
    Overlap links require two shared cameras with distinct centers.

    Args:
        camera_factor_tree: Hierarchy of camera quadratics, pose estimates, and metadata.
        options: Solver and rank settings; measurement_sigma does not reweight supplied factors.

    Returns:
        Camera-only scene, leaf similarities, factor tree, and final camera Bayes tree.

    Raises:
        ValueError: Camera identities, overlap, or geometric rank are insufficient.
        RuntimeError: The root optimizer fails to converge to a finite solution.
    """
    options = options or CameraTreeOptions()
    # Match cameras by filename and align overlapping leaves into an initial global frame.
    leaves = _leaf_inputs(camera_factor_tree)
    initial = _initialize(leaves)
    factor_tree = _assemble_factor_tree(camera_factor_tree, leaves, options)

    # Fix the first leaf's frame and scale before jointly optimizing cameras and similarities.
    graph = gtsam.NonlinearFactorGraph(factor_tree.value.graph)
    graph.push_back(gtsam.NonlinearEqualitySimilarity3(leaves[0].similarity_key, gtsam.Similarity3()))
    optimizer = _optimize_root(graph, initial, options)
    return _camera_tree_result(factor_tree, leaves, graph, initial, optimizer)


def _schur_tree_node(
    value: LocalScene, children: tuple[LocalCameraFactors, ...], *, options: CameraTreeOptions
) -> LocalCameraFactors:
    """Convert a terminal reconstruction while ignoring internal payloads.

    Args:
        value: Node path and optional reconstruction.
        children: Already converted child payloads; only their presence matters here.
        options: Schur-complement production settings.

    Returns:
        Node path and camera-only factors, or None for an internal node.
    """
    path, reconstruction = value
    if children:
        return path, None
    if reconstruction is None:
        raise ValueError(f"Leaf {path}: missing reconstruction.")
    try:
        return path, compute_camera_schur_complement(reconstruction, options)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Leaf {path}: {exc}") from exc


def optimize_camera_tree(scene_tree: SceneTree, options: CameraTreeOptions | None = None) -> CameraTreeResult:
    """Compute leaf Schur complements and merge their frozen camera information without BA.

    Warning:
        SceneTree reconstructs a measurement model from saved geometry; original
        BA factors and gauges are unavailable. Prefer compute_camera_marginal_factors()
        and optimize_camera_factor_tree() to preserve the original BA model.

    Args:
        scene_tree: Hierarchy with named, reconstructed Cal3Bundler cameras and tracks in its leaves.
        options: Producer and solver settings; defaults to one-pixel observation noise.

    Returns:
        Camera-only scene, leaf similarities, factor tree, and final camera Bayes tree.

    Raises:
        ValueError: Camera identities, overlap, observations, or geometric rank are insufficient.
        RuntimeError: The root optimizer fails to converge to a finite solution.
    """
    options = options or CameraTreeOptions()
    warnings.warn(
        "optimize_camera_tree() rebuilds factors from reconstructions and cannot retain the original BA gauge or "
        "noise models. Use compute_camera_marginal_factors() with the original graph and values, then "
        "optimize_camera_factor_tree().",
        UserWarning,
        stacklevel=2,
    )
    camera_factor_tree = scene_tree.map_with_children(partial(_schur_tree_node, options=options))
    return optimize_camera_factor_tree(camera_factor_tree, options)
