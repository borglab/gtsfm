"""Utility functions for merging partitions

Authors: Richi Dubey
"""

from typing import Sequence

import gtsam.noiseModel as noiseModel  # type: ignore
import numpy as np
from gtsam import LevenbergMarquardtOptimizer, NonlinearFactorGraph, Pose3, PriorFactorPose3, Values
from gtsam.symbol_shorthand import X  # type: ignore

KEY = X(0)


def _get_overlapping_pose_pairs(poses1: dict[int, Pose3], poses2: dict[int, Pose3]) -> dict[int, tuple[Pose3, Pose3]]:
    """Returns list of (pose_a, pose_b) tuples for given overlapping keys."""
    keys = [k for k in poses1 if k in poses2]
    if not keys:
        raise ValueError("No overlapping cameras found between the partitions.")
    return {k: (poses1[k], poses2[k]) for k in keys}


def _create_aTb_factors(overlapping_pose_pairs: dict[int, tuple[Pose3, Pose3]]) -> NonlinearFactorGraph:
    """Creates a factor graph with prior factors on aTb from overlapping poses."""
    sigmas: np.ndarray = np.array([0.1] * 3 + [0.1] * 3)  # [rot_x, rot_y, rot_z, tx, ty, tz]
    noise_model = noiseModel.Diagonal.Sigmas(sigmas)
    graph = NonlinearFactorGraph()
    for key, (pose_a, pose_b) in overlapping_pose_pairs.items():
        graph.add(PriorFactorPose3(KEY, pose_a.compose(pose_b.inverse()), noise_model))
    return graph


def _create_aTb_initial_estimate() -> Values:
    """Creates the initial Values object with an identity estimate for aTb."""
    initial = Values()
    initial.insert(KEY, Pose3())  # Use Identity Pose3
    return initial


def _calculate_transform(overlapping_pairs: dict[int, tuple[Pose3, Pose3]]) -> Pose3:
    """Calculates the relative transform aTb between partitions."""
    graph = _create_aTb_factors(overlapping_pairs)
    initial = _create_aTb_initial_estimate()
    try:
        optimizer = LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()
    except Exception as e:
        raise RuntimeError(f"GTSAM optimization failed during merging: {e}") from e
    aTb = result.atPose3(KEY)
    return aTb


def _merge_poses_final(
    poses1: dict[int, Pose3], poses2: dict[int, Pose3], overlapping_keys_set: set[int], aTb_optimized: Pose3
) -> dict[int, Pose3]:
    """Performs the final merge operation using the calculated transform."""
    merged_poses = poses1.copy()
    # Adds transformed non-overlapping poses from poses2 to merged_poses.
    for k, bTi in poses2.items():
        if k not in overlapping_keys_set:
            merged_poses[k] = aTb_optimized.compose(bTi)
    return merged_poses


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
    overlapping_pose_pairs = _get_overlapping_pose_pairs(poses1, poses2)
    aTb = _calculate_transform(overlapping_pose_pairs)
    return _merge_poses_final(poses1, poses2, set(overlapping_pose_pairs.keys()), aTb)
