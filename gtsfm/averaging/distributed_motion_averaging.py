"""Local/global motion averaging on scale-bearing partition reconstructions.

Implements the rotation and translation block structure in Zhu et al., CVPR
2018, Sections 4.4–4.6. Each partition supplies camera poses and relative
camera/landmark measurements in its own coordinate frame. Local solves update
only interior variables. The global solve updates the separators and the
transforms from partition frames to the common world frame.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


class RelativeRotations(NamedTuple):
    """Rotation measurements between pairs of cameras."""

    first: np.ndarray
    second: np.ndarray
    rotation: Rotation


class RelativeDisplacements(NamedTuple):
    """Scale-bearing displacements from a camera to another camera or landmark."""

    first: np.ndarray
    second: np.ndarray
    partition: np.ndarray
    translation: np.ndarray


@dataclass
class MotionState:
    """Camera orientations, entity locations, and partition transforms."""

    camera_owner: np.ndarray
    entity_owner: np.ndarray
    camera_rotation: np.ndarray
    entity_position: np.ndarray
    partition_rotation: np.ndarray
    partition_offset: np.ndarray
    partition_scale: np.ndarray

    def world_rotations(self) -> Rotation:
        """Return camera-to-world orientations, composing only interior cameras.

        Returns:
            Camera orientations in dense camera index order.
        """
        camera = Rotation.from_rotvec(self.camera_rotation).as_matrix()
        interior = self.camera_owner >= 0
        frame = Rotation.from_rotvec(self.partition_rotation).as_matrix()
        camera[interior] = frame[self.camera_owner[interior]] @ camera[interior]
        return Rotation.from_matrix(camera)

    def world_positions(self) -> np.ndarray:
        """Return camera and landmark coordinates in the common frame.

        Returns:
            World coordinates in dense entity index order.
        """
        points = self.entity_position.copy()
        interior = self.entity_owner >= 0
        owner = self.entity_owner[interior]
        frame = Rotation.from_rotvec(self.partition_rotation).as_matrix()[owner]
        points[interior] = (
            self.partition_scale[owner, None] * (frame @ points[interior, :, None]).squeeze(-1)
            + self.partition_offset[owner]
        )
        return points


def _pseudo_huber(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Transform vector residuals to realize the pseudo-Huber objective.

    Args:
        residuals: One Euclidean residual per measured displacement.
        delta: Transition scale in scene units.

    Returns:
        Residuals suitable for a least-squares solver.
    """
    squared_ratio = np.sum(np.square(residuals / delta), axis=1)
    return residuals * np.sqrt(2 / (np.sqrt(1 + squared_ratio) + 1))[:, None]


class _Block(NamedTuple):
    """Three-vector or scalar parameter rows and one dependency set per row."""

    values: np.ndarray
    indices: np.ndarray
    incidence: np.ndarray
    lower_bound: float = -np.inf


def _optimize_blocks(
    residual: Callable[[], np.ndarray],
    blocks: list[_Block],
) -> None:
    """Update parameters in place with a sparse least-squares solve.

    Args:
        residual: Function returning current residuals of shape (num_rows, 3).
        blocks: Parameter arrays, free row indices, and residual-row dependencies.
    """
    active = []
    sparse_rows = []
    sparse_cols = []
    lower_bounds = []
    num_parameters = 0
    for values, indices, incidence, lower_bound in blocks:
        width = values.shape[1]
        if not len(indices):
            continue
        columns = np.full(len(values) + 1, -1, dtype=int)
        columns[indices] = num_parameters + width * np.arange(len(indices))
        # -1 in the incidence array denotes an endpoint with no such parameter.
        row_indices, slots = np.nonzero(columns[incidence] >= 0)
        if not len(row_indices):
            continue
        bases = columns[incidence[row_indices, slots]]
        sparse_rows.append(np.tile(3 * row_indices[:, None] + np.arange(3), (1, width)).ravel())
        sparse_cols.append(np.repeat(bases[:, None] + np.arange(width), 3, axis=1).ravel())
        end = num_parameters + width * len(indices)
        active.append((values, indices, slice(num_parameters, end)))
        lower_bounds.extend([lower_bound] * (end - num_parameters))
        num_parameters = end

    if not active:
        return
    residual_entries, jacobian_rows = np.unique(np.concatenate(sparse_rows), return_inverse=True)
    jacobian = coo_matrix(
        (
            np.ones(len(jacobian_rows), dtype=bool),
            (jacobian_rows, np.concatenate(sparse_cols)),
        ),
        shape=(len(residual_entries), num_parameters),
    ).tocsr()

    def evaluate(parameters: np.ndarray) -> np.ndarray:
        """Evaluate the active residuals after updating the selected rows.

        Args:
            parameters: Packed free parameter vector.

        Returns:
            Flattened residuals affected by this solve.
        """
        for values, indices, section in active:
            values[indices] = parameters[section].reshape(len(indices), values.shape[1])
        return residual().ravel()[residual_entries]

    initial = np.concatenate([values[indices].ravel() for values, indices, _ in active])
    result = least_squares(
        evaluate,
        initial,
        jac_sparsity=jacobian,
        bounds=(lower_bounds, np.inf),
        max_nfev=100,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Motion averaging solve failed: {result.message}")
    # Finite differences leave the underlying arrays at a perturbed point.
    evaluate(result.x)


def average_motions(
    state: MotionState, rotations: RelativeRotations, displacements: RelativeDisplacements, max_rounds: int = 8
) -> dict[str, list[float]]:
    """Alternate local and separator-only rotation, then translation solves.

    Args:
        state: Initial camera, landmark, and partition state, updated in place.
        rotations: Camera-to-camera rotation measurements from local reconstructions.
        displacements: Scale-bearing position measurements from local reconstructions.
        max_rounds: Maximum rounds per phase. Stop after under 1% RMS improvement.

    Returns:
        Rotation and translation residual RMS before and after each round.
    """
    i = displacements.first
    j = displacements.second
    source = displacements.partition
    num_partitions = len(state.partition_rotation)
    partitions = np.arange(num_partitions)
    camera_indices = np.arange(len(state.camera_owner))
    entity_indices = np.arange(len(state.entity_owner))
    boundary_cameras = camera_indices[state.camera_owner < 0]
    if not len(boundary_cameras):
        raise ValueError("At least one separator camera is required")
    anchor = boundary_cameras[0]
    rotation_cameras = np.column_stack((rotations.first, rotations.second))
    rotation_partitions = state.camera_owner[rotation_cameras]
    entity_edges = np.stack((i, j), axis=1)
    entity_partitions = state.entity_owner[entity_edges]

    def rotation_residual() -> np.ndarray:
        """Calculate SO(3) log residuals for the camera-to-camera measurements.

        Returns:
            Angular residual vectors, in radians.
        """
        world = state.world_rotations()
        predicted = world[rotations.second].inv() * world[rotations.first]
        return (rotations.rotation.inv() * predicted).as_rotvec()

    rotation_trace = [float(np.sqrt(np.mean(rotation_residual() ** 2)))]
    for _ in range(max_rounds):
        for p in partitions:
            local = camera_indices[state.camera_owner == p]
            _optimize_blocks(
                rotation_residual,
                [_Block(state.camera_rotation, local, rotation_cameras)],
            )
        free_transforms = partitions[1:]
        _optimize_blocks(
            rotation_residual,
            [
                _Block(state.camera_rotation, boundary_cameras[1:], rotation_cameras),
                _Block(state.partition_rotation, free_transforms, rotation_partitions),
            ],
        )
        rotation_trace.append(float(np.sqrt(np.mean(rotation_residual() ** 2))))
        if rotation_trace[-1] >= 0.99 * rotation_trace[-2]:
            break

    world_displacement = state.world_rotations()[i].apply(displacements.translation)
    scale_partitions = np.column_stack((entity_partitions, source))

    def translation_residual() -> np.ndarray:
        """Calculate scale-aware displacement errors with orientations held fixed.

        Returns:
            One position error vector per camera or landmark measurement.
        """
        world = state.world_positions()
        return world[j] - world[i] - state.partition_scale[source, None] * world_displacement

    def robust_translation_residual() -> np.ndarray:
        """Return pseudo-Huber weighted position residuals.

        Returns:
            One least-squares residual per measured displacement.
        """
        return _pseudo_huber(translation_residual(), delta=0.1)

    translation_trace = [float(np.sqrt(np.mean(translation_residual() ** 2)))]
    boundary = entity_indices[(state.entity_owner < 0) & (entity_indices != anchor)]
    active_transforms = np.unique(state.entity_owner[state.entity_owner >= 0])
    for _ in range(max_rounds):
        for p in partitions:
            local = entity_indices[state.entity_owner == p]
            _optimize_blocks(
                robust_translation_residual,
                [_Block(state.entity_position, local, entity_edges)],
            )
        _optimize_blocks(
            robust_translation_residual,
            [
                _Block(state.entity_position, boundary, entity_edges),
                _Block(state.partition_offset, active_transforms, entity_partitions),
                _Block(state.partition_scale[:, None], partitions[1:], scale_partitions, lower_bound=1e-8),
            ],
        )
        translation_trace.append(float(np.sqrt(np.mean(translation_residual() ** 2))))
        if translation_trace[-1] >= 0.99 * translation_trace[-2]:
            break

    return {"rotation_rms_rad": rotation_trace, "translation_rms_units": translation_trace}
