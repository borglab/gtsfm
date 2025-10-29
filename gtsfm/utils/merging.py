"""Utility functions for merging clusters

Authors: Richi Dubey
"""

import gtsam.noiseModel as noiseModel  # type: ignore
import numpy as np
from gtsam import Similarity3  # type: ignore
from gtsam import LevenbergMarquardtOptimizer, NonlinearFactorGraph, Pose3, PriorFactorPose3, Values
from gtsam.symbol_shorthand import X  # type: ignore

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.types import create_camera

KEY = X(0)


def _create_aTb_factors(a: dict[int, Pose3], b: dict[int, Pose3], common_keys: list[int]) -> NonlinearFactorGraph:
    """Creates a factor graph with prior factors on aTb from overlapping poses."""
    sigmas: np.ndarray = np.array([0.1] * 3 + [0.1] * 3)  # [rot_x, rot_y, rot_z, tx, ty, tz]
    noise_model = noiseModel.Diagonal.Sigmas(sigmas)
    graph = NonlinearFactorGraph()
    for i in common_keys:
        aTi = a[i]
        bTi = b[i]
        graph.add(PriorFactorPose3(KEY, aTi * bTi.inverse(), noise_model))
    return graph


def _create_aTb_initial_estimate(a: dict[int, Pose3], b: dict[int, Pose3], common_keys: list[int]) -> Values:
    """Creates the initial Values object with an identity estimate for aTb."""
    initial = Values()
    i = common_keys[0]
    aTi = a[i]
    bTi = b[i]
    initial.insert(KEY, aTi * bTi.inverse())
    return initial


def calculate_transform(a: dict[int, Pose3], b: dict[int, Pose3]) -> Pose3:
    """Calculates the relative transform aTb between partitions."""
    common_keys = [i for i in a if i in b]
    if not common_keys:
        raise ValueError("No overlapping cameras found between the partitions.")
    graph = _create_aTb_factors(a, b, common_keys)
    initial = _create_aTb_initial_estimate(a, b, common_keys)
    try:
        optimizer = LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()
    except Exception as e:
        raise RuntimeError(f"GTSAM optimization failed during merging: {e}") from e
    aTb = result.atPose3(KEY)
    return aTb


def add_transformed_poses(a: dict[int, Pose3], aTb: Pose3, b: dict[int, Pose3]) -> dict[int, Pose3]:
    """Performs the final merge operation using the calculated transform."""
    merged_poses = a.copy()
    # Adds transformed non-overlapping poses from b to merged_poses.
    for i, bTi in b.items():
        if i not in merged_poses:
            merged_poses[i] = aTb * bTi
    return merged_poses


def merge_two_pose_maps(a: dict[int, Pose3], b: dict[int, Pose3]) -> dict[int, Pose3]:
    """
    Merges poses from two partitions by finding and applying relative transform aTb.

    Assumes a are relative to frame 'a' and b are relative to frame 'b'.
    Finds 'aTb' (from frame 'b' to frame 'a') via overlapping poses.
    Transforms non-overlapping poses from partition 2 into frame 'a' and merges.

    Args:
        a: dictionary {camera_index: pose_in_frame_a}.
        b: dictionary {camera_index: pose_in_frame_b}.

    Returns:
        A merged dictionary {camera_index: pose_in_frame_a}.

    Raises:
        ValueError: If no overlapping cameras are found between the two partitions.
        RuntimeError: If GTSAM optimization fails.
    """
    aTb = calculate_transform(a, b)
    return add_transformed_poses(a, aTb, b)


def estimate_sim3_old(a: dict[int, Pose3], b: dict[int, Pose3]) -> Similarity3:
    """Estimate the similarity transform, using very simple scale estimator."""

    aTb = calculate_transform(a, b)

    # poor man scale estimation
    common_keys = [i for i in a if i in b]
    i, j = common_keys[0], common_keys[-1]
    b_norm = np.linalg.norm(b[i].translation() - b[j].translation())
    a_norm = np.linalg.norm(a[i].translation() - a[j].translation())
    scale = float(a_norm / b_norm)

    return Similarity3(aTb.rotation().matrix(), aTb.translation(), scale)


def estimate_sim3(a: dict[int, Pose3], b: dict[int, Pose3]) -> Similarity3:
    """Estimate the similarity transform using GTSAM."""
    common_keys = [i for i in a if i in b]
    pose_pairs = [(a[i], b[i]) for i in common_keys]
    try:
        return Similarity3.Align(pose_pairs)
    except Exception as e:
        raise RuntimeError(
            f"merging.estimate_sim3: Similarity3.Align failed to estimate similarity transform "
            f"with {len(pose_pairs)} pose pairs: {e}."
        ) from e


def merge(a: GtsfmData, b: GtsfmData) -> GtsfmData:
    """Merge another GtsfmData into this one in-place.

    Args:
        b: GtsfmData to merge into this one.

    Returns:
        Merged GtsfmData, a new object. Coordinate frame A is used as prime.
    """
    aSb = estimate_sim3(a.poses(), b.poses())

    # Create merged cameras with updated poses. Only b-poses need to be updated.
    merged_cameras = a.cameras().copy()
    for i, cam in b.cameras().items():
        # TODO: what to do if we have conflicting calibrations?
        if i not in merged_cameras:
            bTi = cam.pose()
            merged_cameras[i] = create_camera(aSb.transformFrom(bTi), cam.calibration())

    # Create merged tracks
    merged_tracks = a.tracks().copy()
    # For all b_tracks, update the point by multiplying with aSb:
    for track in b.tracks():
        track.p = aSb.transformFrom(track.p)  # from b to a
        merged_tracks.append(track)

    # Create merged GtsfmData
    # TODO: Check whether we need to remap tracks
    merged_data = GtsfmData(len(merged_cameras), merged_cameras, [])
    merged_data._tracks = merged_tracks

    return merged_data
