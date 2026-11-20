"""Regression for separator constraints and scale-bearing translation averaging."""

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gtsfm.averaging.distributed_motion_averaging import (
    MotionState,
    RelativeDisplacements,
    RelativeRotations,
    average_motions,
)
from scripts.run_distributed_motion_averaging import _prepare_frontend


def test_alternating_solves_recover_scaled_camera_and_landmark_constraints() -> None:
    """Correct perturbed rotations, scales, and centers across two local frames."""
    camera_owner = np.array([0, -1, -1, 1])
    entity_owner = np.array([0, -1, -1, 1, 0, 1, -1])
    orientation = Rotation.from_rotvec([[0.03, 0.1, -0.2], [0.1, 0.2, 0], [-0.1, 0.3, 0.2], [0.3, -0.5, 0.5]])
    world = np.array([[1, 0, 0], [1, 1, 0], [1, 2, 0.2], [1.2, 1.3, 1.9], [3, 0, 1.0], [0.2, 1.7, 1.0], [0, 1, 3.0]])
    scale = np.array([1.0, 0.7])
    i = np.array([0, 0, 1, 1, 1, 2, 0, 1, 3, 2, 1, 2])
    j = np.array([1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    source = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    camera_i, camera_j = i[:6], j[:6]
    rotations = RelativeRotations(camera_i, camera_j, orientation[camera_j].inv() * orientation[camera_i])
    displacements = RelativeDisplacements(
        i,
        j,
        source,
        orientation[i].inv().apply(world[j] - world[i]) / scale[source, None],
    )
    partition_rotation = np.array([[0, 0, 0], [0.2, -0.3, 0.1]])
    partition_offset = np.array([[0, 0, 0], [-0.1, 0.4, 0.8]])
    frame_from_world = Rotation.from_rotvec(partition_rotation[1]).inv()
    local_rotations = orientation.as_rotvec()
    local_rotations[3] = (frame_from_world * orientation[3]).as_rotvec()
    local_positions = world.copy()
    local_positions[[3, 5]] = frame_from_world.apply(world[[3, 5]] - partition_offset[1]) / scale[1]
    state = MotionState(
        camera_owner, entity_owner, local_rotations, local_positions, partition_rotation, partition_offset, scale.copy()
    )
    state.camera_rotation[[0, 2, 3]] += 0.05
    state.entity_position[[0, 2, 3, 4, 5, 6]] += 0.08
    state.partition_scale[1] = 0.85

    residuals = average_motions(state, rotations, displacements, max_rounds=40)

    assert residuals["rotation_rms_rad"][-1] < 0.05 * residuals["rotation_rms_rad"][0]
    assert residuals["translation_rms_units"][-1] < 0.1 * residuals["translation_rms_units"][0]
    np.testing.assert_allclose(state.partition_scale, scale, atol=0.02)
    np.testing.assert_allclose((state.world_rotations().inv() * orientation).magnitude(), 0, atol=0.01)
    np.testing.assert_allclose(state.world_positions()[:4], world[:4], atol=0.04)


def test_snapshot_captures_uncheckpointed_verified_matches(tmp_path: Path) -> None:
    """Freeze the current verified pairs while preserving the source database.

    Args:
        tmp_path: Directory isolated from the source test dataset.
    """
    fixture = Path(__file__).resolve().parents[1] / "data" / "set1_lund_door" / "colmap.db"
    source = tmp_path / "source.db"
    shutil.copyfile(fixture, source)
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        initial_bytes = source.read_bytes()
        _, original_graph = _prepare_frontend(source, tmp_path / "original.db", 15)
        with writer:
            writer.execute(
                "DELETE FROM two_view_geometries WHERE pair_id = "
                "(SELECT min(pair_id) FROM two_view_geometries WHERE rows >= 15)"
            )
        assert source.read_bytes() == initial_bytes
        snapshot = tmp_path / "changed.db"
        _, changed_graph = _prepare_frontend(source, snapshot, 15)
        assert changed_graph.number_of_edges() == original_graph.number_of_edges() - 1
        assert source.read_bytes() == initial_bytes
