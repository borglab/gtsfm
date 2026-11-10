"""Fuse frozen leaf BA information in a common frame without retaining landmarks.

Each leaf contributes a camera-only Gaussian Bayes tree in its own coordinate
gauge. Similarity-aware factors carry its full joint camera information into a
single root optimization. The result approximates BA at the supplied leaf
linearizations; no reprojection factors are evaluated after condensation.
"""

from __future__ import annotations

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
    """Settings for fixed-intrinsic, frozen-linearization camera fusion."""

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
class LeafCameraSummary:
    """Landmark-free leaf model; reference poses are keyed by global X(i)."""

    path: Path
    similarity_key: int
    local_to_global: dict[int, int]
    reference_poses: gtsam.Values
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
    scene: GtsfmData
    local_to_global: dict[int, int]
    similarity_key: int


def _leaf_inputs(scene_tree: SceneTree) -> tuple[_LeafInput, ...]:
    """Resolve camera identity by filename, inspecting only terminal nodes.

    Args:
        scene_tree: Tree of optional reconstructions.

    Returns:
        Validated leaf inputs in deterministic tree traversal order.
    """
    names: set[str] = set()
    paths: set[Path] = set()
    for node in scene_tree.leaves():
        path, scene = node.value
        if path in paths:
            raise ValueError(f"Duplicate leaf path: {path}.")
        paths.add(path)
        if scene is None or len(scene.get_valid_camera_indices()) < 2:
            raise ValueError(f"Leaf {path}: need at least two reconstructed cameras.")
        local_names: set[str] = set()
        for i in scene.get_valid_camera_indices():
            name = scene.get_image_info(i).name
            if not name or name in local_names:
                raise ValueError(f"Leaf {path}: missing or duplicate camera filename at index {i}.")
            camera = scene.get_camera(i)
            if camera is None or not isinstance(camera.calibration(), gtsam.Cal3Bundler):
                raise ValueError(f"Leaf {path}: only Cal3Bundler cameras are supported.")
            if not np.all(np.isfinite(camera.pose().matrix())):
                raise ValueError(f"Leaf {path}: nonfinite camera pose at index {i}.")
            local_names.add(name)
        names.update(local_names)
    global_indices = {name: i for i, name in enumerate(sorted(names))}
    inputs = []
    for leaf_index, node in enumerate(scene_tree.leaves()):
        path, scene = node.value
        assert scene is not None
        mapping = {i: global_indices[scene.get_image_info(i).name] for i in scene.get_valid_camera_indices()}
        inputs.append(_LeafInput(path, scene, mapping, S(leaf_index)))
    return tuple(inputs)


def _local_poses(leaf: _LeafInput) -> dict[int, gtsam.Pose3]:
    """Get local poses indexed by their global camera identities.

    Args:
        leaf: Reconstruction with resolved camera identities.

    Returns:
        Global camera indices mapped to poses in the leaf frame.
    """
    return {global_i: leaf.scene.get_camera(i).pose() for i, global_i in leaf.local_to_global.items()}


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
        leaves: Validated leaf scenes with global camera identities.

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


def _linearize_leaf(leaf: _LeafInput, options: CameraTreeOptions) -> tuple[gtsam.GaussianFactorGraph, gtsam.Values]:
    """Linearize fixed-calibration observations once in the leaf's local frame.

    Args:
        leaf: Reconstruction and camera identity mapping.
        options: Measurement noise settings.

    Returns:
        Linearized graph and reference camera poses.
    """
    values = gtsam.Values()
    references = gtsam.Values()
    graph = gtsam.NonlinearFactorGraph()
    noise = gtsam.noiseModel.Isotropic.Sigma(2, options.measurement_sigma)
    for i, global_i in leaf.local_to_global.items():
        camera = leaf.scene.get_camera(i)
        assert camera is not None
        calibration = camera.calibration()
        # The PinholePose constructor requires an owned/shared calibration object.
        held_calibration = gtsam.Cal3Bundler(
            calibration.fx(), calibration.k1(), calibration.k2(), calibration.px(), calibration.py()
        )
        values.insert(X(global_i), gtsam.PinholePoseCal3Bundler(camera.pose(), held_calibration))
        references.insert(X(global_i), camera.pose())
    for j, track in enumerate(leaf.scene.tracks()):
        measurements = [track.measurement(k) for k in range(track.numberMeasurements())]
        measurements = [(i, uv) for i, uv in measurements if i in leaf.local_to_global]
        if len({i for i, _ in measurements}) < 2:
            continue
        values.insert(L(j), track.point3())
        for i, uv in measurements:
            graph.push_back(gtsam.GeneralSFMFactorPoseCal3Bundler(uv, noise, X(leaf.local_to_global[i]), L(j)))
    if graph.empty():
        raise ValueError(f"Leaf {leaf.path}: no tracks with at least two camera observations.")
    return graph.linearize(values), references


def _add_coordinate_gauge(graph: gtsam.GaussianFactorGraph, references: gtsam.Values) -> None:
    """Fix one local pose and one baseline direction using exact coordinate constraints.

    Args:
        graph: Camera graph to augment with seven constrained rows.
        references: Local reference poses, keyed by global camera keys.
    """
    keys = list(references.keys())
    anchor_key = keys[0]
    anchor = references.atPose3(anchor_key)
    baseline_key = max(
        keys[1:], key=lambda key: np.linalg.norm(references.atPose3(key).translation() - anchor.translation())
    )
    baseline_pose = references.atPose3(baseline_key)
    direction = baseline_pose.translation() - anchor.translation()
    direction /= np.linalg.norm(direction)
    graph.push_back(gtsam.JacobianFactor(anchor_key, np.eye(6), np.zeros(6), gtsam.noiseModel.Constrained.All(6)))
    anchor_row = np.zeros((1, 6))
    baseline_row = np.zeros((1, 6))
    anchor_row[0, 3:] = -direction @ anchor.rotation().matrix()
    baseline_row[0, 3:] = direction @ baseline_pose.rotation().matrix()
    graph.push_back(
        gtsam.JacobianFactor(
            anchor_key, anchor_row, baseline_key, baseline_row, np.zeros(1), gtsam.noiseModel.Constrained.All(1)
        )
    )


def _lift_conditional(
    conditional: gtsam.GaussianFactor, references: gtsam.Values, similarity_key: int
) -> gtsam.CustomFactor:
    """Evaluate a camera conditional through a global-to-leaf similarity.

    Args:
        conditional: Gaussian camera conditional with six columns per camera.
        references: Frozen local reference poses.
        similarity_key: Key for the global-to-leaf Similarity3 variable.

    Returns:
        Nonlinear factor preserving the conditional's original noise model.
    """
    factor = gtsam.JacobianFactor(conditional)
    keys = factor.keys()
    matrix = np.array(factor.getA())
    rhs = np.array(factor.getb())
    reference_poses = tuple(references.atPose3(key) for key in keys)

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
        for column, (key, reference) in enumerate(zip(keys, reference_poses)):
            pose = values.atPose3(key)
            if jacobians is None:
                deltas.append(reference.localCoordinates(similarity.transformFrom(pose)))
                continue
            h_similarity = np.zeros((6, 7), order="F")
            h_pose = np.zeros((6, 6), order="F")
            h_local = np.zeros((6, 6), order="F")
            local_pose = similarity.transformFrom(pose, h_similarity, h_pose)
            deltas.append(reference.localCoordinates(local_pose, None, h_local))
            block = matrix[:, 6 * column : 6 * (column + 1)] @ h_local
            jacobians[0] = jacobians[0] + block @ h_similarity
            jacobians[column + 1] = block @ h_pose
        return matrix @ np.concatenate(deltas) - rhs

    return gtsam.CustomFactor(factor.get_model(), [similarity_key, *keys], error)


def _condense_leaf(leaf: _LeafInput, options: CameraTreeOptions) -> LeafCameraSummary:
    """Eliminate all structure and turn the camera Bayes tree into lifted factors.

    Args:
        leaf: Validated leaf reconstruction.
        options: Noise and geometric rank settings.

    Returns:
        A summary retaining camera information and no landmark variables.
    """
    linear_graph, references = _linearize_leaf(leaf, options)
    try:
        # Keep complete camera factors, including residual offsets; Marginals returns only matrices in Python.
        camera_graph = linear_graph.marginal(list(references.keys()), gtsam.EliminateQR)
        if set(camera_graph.keyVector()) != set(references.keys()):
            raise ValueError("Some cameras have no retained observations.")
        matrix, _ = camera_graph.jacobian()
        column_norms = np.linalg.norm(matrix, axis=0)
        scaled = matrix / np.where(column_norms > 0, column_norms, 1)
        singular_values = np.linalg.svd(scaled, compute_uv=False)
        rank = np.count_nonzero(singular_values > options.rank_tolerance * singular_values[0])
        expected_rank = 6 * references.size() - 7
        if rank < expected_rank:
            raise ValueError(f"Insufficient geometric rank: {rank}, expected {expected_rank} after removing gauge.")
        _add_coordinate_gauge(camera_graph, references)
        bayes_tree = camera_graph.eliminateMultifrontal(
            gtsam.Ordering.ColamdGaussianFactorGraph(camera_graph), gtsam.EliminateQR
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Leaf {leaf.path}: camera condensation failed: {exc}") from exc
    conditionals = gtsam.GaussianFactorGraph(bayes_tree)
    factors = tuple(
        _lift_conditional(conditionals.at(k), references, leaf.similarity_key) for k in range(conditionals.size())
    )
    return LeafCameraSummary(leaf.path, leaf.similarity_key, leaf.local_to_global, references, bayes_tree, factors)


def _assemble_subtree(
    value: LocalScene,
    children: tuple[CameraSubtreeSummary, ...],
    *,
    inputs_by_path: dict[Path, _LeafInput],
    options: CameraTreeOptions,
) -> CameraSubtreeSummary:
    """Combine child graphs once, or condense a terminal reconstruction.

    Args:
        value: Node path and optional reconstruction payload.
        children: Summaries already assembled for the immediate children.
        inputs_by_path: Validated inputs for terminal nodes.
        options: Leaf condensation settings.

    Returns:
        All factors and leaf summaries belonging to this subtree.
    """
    graph = gtsam.NonlinearFactorGraph()
    if children:
        # Internal nodes collect each child's evidence once, ignoring their own reconstruction.
        for child in children:
            graph.push_back(child.graph)
        return CameraSubtreeSummary(graph, tuple(leaf for child in children for leaf in child.leaves))
    # Eliminate this leaf's landmarks and express its camera evidence through a similarity.
    summary = _condense_leaf(inputs_by_path[value[0]], options)
    for factor in summary.factors:
        graph.push_back(factor)
    return CameraSubtreeSummary(graph, (summary,))


def _assemble_factor_tree(
    scene_tree: SceneTree, leaves: tuple[_LeafInput, ...], options: CameraTreeOptions
) -> Tree[CameraSubtreeSummary]:
    """Assemble all leaf factors bottom-up while preserving the input hierarchy.

    Args:
        scene_tree: Hierarchy to traverse.
        leaves: Validated leaf reconstructions.
        options: Leaf condensation settings.

    Returns:
        Mirrored tree containing the accumulated camera factors at each node.
    """
    assemble = partial(_assemble_subtree, inputs_by_path={leaf.path: leaf for leaf in leaves}, options=options)
    return scene_tree.map_with_children(assemble)


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
    # Factor the remaining camera quadratic into the output Gaussian Bayes tree.
    return final_camera_graph.eliminateMultifrontal(
        gtsam.Ordering.ColamdGaussianFactorGraph(final_camera_graph), gtsam.EliminateQR
    )


def _camera_scene(leaves: tuple[_LeafInput, ...], optimized: gtsam.Values) -> GtsfmData:
    """Build a camera-only scene from optimized poses and source metadata.

    Args:
        leaves: Leaf reconstructions providing calibration and image metadata.
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
            source = leaf.scene.get_camera(local_i)
            info = leaf.scene.get_image_info(local_i)
            scene.add_camera(
                global_i, gtsam.PinholeCameraCal3Bundler(optimized.atPose3(X(global_i)), source.calibration())
            )
            scene.set_image_info(global_i, name=info.name, shape=info.shape)
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


def optimize_camera_tree(scene_tree: SceneTree, options: CameraTreeOptions | None = None) -> CameraTreeResult:
    """Fuse frozen leaf camera information in the first leaf's frame without further BA.

    Intrinsics stay fixed, and internal reconstruction payloads are ignored.
    Overlap links require two shared cameras with distinct centers.

    Args:
        scene_tree: Hierarchy with named, reconstructed Cal3Bundler cameras and tracks in its leaves.
        options: Numerical settings; defaults to one-pixel observation noise.

    Returns:
        Camera-only scene, leaf similarities, factor tree, and final camera Bayes tree.

    Raises:
        ValueError: Camera identities, overlap, or geometric rank are insufficient.
        RuntimeError: The root optimizer fails to converge to a finite solution.
    """
    options = options or CameraTreeOptions()
    # Match cameras by filename and align overlapping leaves into an initial global frame.
    leaves = _leaf_inputs(scene_tree)
    initial = _initialize(leaves)
    factor_tree = _assemble_factor_tree(scene_tree, leaves, options)

    # Fix the first leaf's frame and scale before jointly optimizing cameras and similarities.
    graph = gtsam.NonlinearFactorGraph(factor_tree.value.graph)
    graph.push_back(gtsam.NonlinearEqualitySimilarity3(leaves[0].similarity_key, gtsam.Similarity3()))
    optimizer = _optimize_root(graph, initial, options)
    return _camera_tree_result(factor_tree, leaves, graph, initial, optimizer)
