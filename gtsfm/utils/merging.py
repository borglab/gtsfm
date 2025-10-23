"""Utility functions for merging partitions

Authors: Richi Dubey
"""

import gtsam.noiseModel as noiseModel  # type: ignore
import numpy as np
from gtsam import LevenbergMarquardtOptimizer, NonlinearFactorGraph, Pose3, PriorFactorPose3, Symbol, Values

import gtsfm.utils.merging as merging_utils


def _get_overlapping_keys(poses1: dict[int, Pose3], poses2: dict[int, Pose3]) -> list[int]:
    """Returns keys present in both pose dictionaries, raising ValueError if none."""
    keys = [k for k in poses1 if k in poses2]
    if not keys:
        raise ValueError("No overlapping cameras found between the partitions.")
    return keys


def _get_overlapping_pose_pairs(
    poses1: dict[int, Pose3], poses2: dict[int, Pose3], keys: list[int]
) -> list[tuple[Pose3, Pose3]]:
    """Returns list of (pose_a, pose_b) tuples for given overlapping keys."""
    return [(poses1[k], poses2[k]) for k in keys]


def _get_overlap_data(
    poses1: dict[int, Pose3], poses2: dict[int, Pose3]
) -> tuple[list[int], list[tuple[Pose3, Pose3]]]:
    """Gets keys and pose pairs for overlapping cameras."""
    keys = _get_overlapping_keys(poses1, poses2)
    pairs = _get_overlapping_pose_pairs(poses1, poses2, keys)
    return keys, pairs


def _get_noise_model() -> noiseModel.Base:
    """Defines the noise model for the prior factors."""
    # Using Diagonal noise based on previous debugging, tune if needed.
    sigmas: np.ndarray = np.array([0.1] * 3 + [0.1] * 3)  # [rot_x, rot_y, rot_z, tx, ty, tz]
    return noiseModel.Diagonal.Sigmas(sigmas)


def _create_aTb_factors(
    overlapping_pose_pairs: list[tuple[Pose3, Pose3]],
    aTb_key: Symbol,
    noise_model: noiseModel.Base,
) -> NonlinearFactorGraph:
    """Creates a factor graph with prior factors on aTb from overlapping poses."""
    graph = NonlinearFactorGraph()
    for pose_a, pose_b in overlapping_pose_pairs:
        graph.add(PriorFactorPose3(aTb_key.key(), pose_a.compose(pose_b.inverse()), noise_model))
    return graph


def _create_aTb_initial_estimate(aTb_key: Symbol) -> Values:
    """Creates the initial Values object with an identity estimate for aTb."""
    initial = Values()
    initial.insert(aTb_key.key(), Pose3())  # Use Identity Pose3
    return initial


def _prepare_graph_and_initial(
    overlapping_pairs: list[tuple[Pose3, Pose3]], aTb_key: Symbol
) -> tuple[NonlinearFactorGraph, Values]:
    """Prepare graph and initial values for optimization."""
    noise = _get_noise_model()
    graph = _create_aTb_factors(overlapping_pairs, aTb_key, noise)
    initial = _create_aTb_initial_estimate(aTb_key)
    return graph, initial


def _run_optimization(graph: NonlinearFactorGraph, initial: Values) -> Values:
    """Runs the optimizer and returns the result values."""
    optimizer = LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    return result


def _optimize_graph_safe(graph: NonlinearFactorGraph, initial: Values) -> Values:
    """Optimizes graph, wrapping optimization call in try-except."""
    try:
        return _run_optimization(graph, initial)
    except Exception as e:
        raise RuntimeError(f"GTSAM optimization failed during merging: {e}") from e


def _extract_optimized_aTb(result: Values, aTb_key: Symbol) -> Pose3:
    """Extracts the optimized Pose3 transformation from the Values object."""
    aTb_optimized = result.atPose3(aTb_key.key())
    return aTb_optimized


def _run_and_extract_transform(graph: NonlinearFactorGraph, initial: Values, aTb_key: Symbol) -> Pose3:
    """Run optimization and extract the resulting transform."""
    result = _optimize_graph_safe(graph, initial)
    aTb = _extract_optimized_aTb(result, aTb_key)
    return aTb


def _calculate_transform(overlapping_pairs: list[tuple[Pose3, Pose3]]) -> Pose3:
    """Calculates the relative transform aTb between partitions."""
    aTb_key = Symbol("x", 0)  # Define the key for the transform
    graph, initial = _prepare_graph_and_initial(overlapping_pairs, aTb_key)
    aTb = _run_and_extract_transform(graph, initial, aTb_key)
    return aTb


def _transform_and_add_new_poses(
    merged_poses: dict[int, Pose3],
    poses2: dict[int, Pose3],
    overlapping_keys_set: set[int],
    aTb_optimized: Pose3,
) -> None:
    """Adds transformed non-overlapping poses from poses2 to merged_poses."""
    for k, bTi in poses2.items():
        if k not in overlapping_keys_set:
            merged_poses[k] = aTb_optimized.compose(bTi)


def _merge_poses_final(
    poses1: dict[int, Pose3],
    poses2: dict[int, Pose3],
    overlapping_keys: list[int],
    aTb_optimized: Pose3,
) -> dict[int, Pose3]:
    """Performs the final merge operation using the calculated transform."""
    merged = poses1.copy()
    _transform_and_add_new_poses(merged, poses2, set(overlapping_keys), aTb_optimized)
    return merged


def merge_two_pose_maps(poses1: dict[int, Pose3], poses2: dict[int, Pose3]) -> dict[int, Pose3]:
    """
    Merges poses from two partitions by finding and applying relative transform aTb.

    Assumes poses1 are relative to frame 'a' and poses2 are relative to frame 'b'.
    Finds 'aTb' (from frame 'b' to frame 'a') via overlapping poses.
    Transforms non-overlapping poses from partition 2 into frame 'a' and merges.

    Args:
        poses1: dictionary {camera_index: pose_in_frame_a}.
        poses2: dictionary {camera_index: pose_in_frame_b}.

    Returns:
        A merged dictionary {camera_index: pose_in_frame_a}.

    Raises:
        ValueError: If no overlapping cameras are found between the two partitions.
        RuntimeError: If GTSAM optimization fails.
    """
    keys, pairs = merging_utils._get_overlap_data(poses1, poses2)
    aTb = merging_utils._calculate_transform(pairs)
    return merging_utils._merge_poses_final(poses1, poses2, keys, aTb)
