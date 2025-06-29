"""Utility functions for merging partitions

Authors: Richi Dubey
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from gtsam import (
    NonlinearFactorGraph,
    Values,
    Symbol,
    noiseModel,
    PriorFactorPose3,
    LevenbergMarquardtOptimizer,
    Pose3,
)


def _get_overlapping_keys(poses1: Dict[int, Pose3], poses2: Dict[int, Pose3]) -> List[int]:
    """Returns keys present in both pose dictionaries, raising ValueError if none."""
    keys = [k for k in poses1 if k in poses2]
    if not keys:
        raise ValueError("No overlapping cameras found between the partitions.")
    return keys


def _get_overlapping_pose_pairs(
    poses1: Dict[int, Pose3], poses2: Dict[int, Pose3], keys: List[int]
) -> List[Tuple[Pose3, Pose3]]:
    """Returns list of (pose_a, pose_b) tuples for given overlapping keys."""
    return [(poses1[k], poses2[k]) for k in keys]


def _get_overlap_data(
    poses1: Dict[int, Pose3], poses2: Dict[int, Pose3]
) -> Tuple[List[int], List[Tuple[Pose3, Pose3]]]:
    """Gets keys and pose pairs for overlapping cameras."""
    keys = _get_overlapping_keys(poses1, poses2)
    pairs = _get_overlapping_pose_pairs(poses1, poses2, keys)
    return keys, pairs


def _get_noise_model() -> noiseModel.Base:
    """Defines the noise model for the prior factors."""
    # Using Diagonal noise based on previous debugging, tune if needed.
    sigmas = np.array([0.1] * 3 + [0.1] * 3)  # [rot_x, rot_y, rot_z, tx, ty, tz]
    return noiseModel.Diagonal.Sigmas(sigmas)


def _create_aTb_factors(
    overlapping_pose_pairs: List[Tuple[Pose3, Pose3]],
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
    overlapping_pairs: List[Tuple[Pose3, Pose3]], aTb_key: Symbol
) -> Tuple[NonlinearFactorGraph, Values]:
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
    print(f"Optimized aTb (from partition 2 frame to 1 frame):\n{aTb_optimized}")
    return aTb_optimized


def _run_and_extract_transform(graph: NonlinearFactorGraph, initial: Values, aTb_key: Symbol) -> Pose3:
    """Run optimization and extract the resulting transform."""
    result = _optimize_graph_safe(graph, initial)
    aTb = _extract_optimized_aTb(result, aTb_key)
    return aTb


def _calculate_transform(overlapping_pairs: List[Tuple[Pose3, Pose3]]) -> Pose3:
    """Calculates the relative transform aTb between partitions."""
    aTb_key = Symbol("x", 0)  # Define the key for the transform
    graph, initial = _prepare_graph_and_initial(overlapping_pairs, aTb_key)
    aTb = _run_and_extract_transform(graph, initial, aTb_key)
    return aTb


def _transform_and_add_new_poses(
    merged_poses: Dict[int, Pose3],
    poses2: Dict[int, Pose3],
    overlapping_keys_set: Set[int],
    aTb_optimized: Pose3,
) -> None:
    """Adds transformed non-overlapping poses from poses2 to merged_poses."""
    for k, bTi in poses2.items():
        if k not in overlapping_keys_set:
            merged_poses[k] = aTb_optimized.compose(bTi)


def _merge_poses_final(
    poses1: Dict[int, Pose3],
    poses2: Dict[int, Pose3],
    overlapping_keys: List[int],
    aTb_optimized: Pose3,
) -> Dict[int, Pose3]:
    """Performs the final merge operation using the calculated transform."""
    merged = poses1.copy()
    _transform_and_add_new_poses(merged, poses2, set(overlapping_keys), aTb_optimized)
    return merged
