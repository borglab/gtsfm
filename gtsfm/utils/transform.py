"""Utility functions for transporting geometry between coordinate frames.

Authors: Ayush Baid, John Lambert, Frank Dellaert
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from gtsam import Pose3, Rot3, SfmTrack, Similarity3  # type: ignore

from gtsfm.common.types import CAMERA_TYPE, create_camera


def Rot3s_with_so3(rotations_b: Sequence[Rot3], aRb: Rot3) -> List[Rot3]:
    """Transport a list of Rot3s from frame ``b`` to frame ``a`` using an SO(3) transform."""
    return [aRb.compose(rotation_b) for rotation_b in rotations_b]


def optional_Rot3s_with_so3(rotations_b: Sequence[Optional[Rot3]], aRb: Rot3) -> List[Optional[Rot3]]:
    """Transport an optional rotation list from frame ``b`` to frame ``a`` using an SO(3) transform."""
    return [aRb.compose(rotation_b) if rotation_b is not None else None for rotation_b in rotations_b]


def Pose3_map_with_se3(pose_map_b: Mapping[int, Pose3], aTb: Pose3) -> Dict[int, Pose3]:
    """Transport a Pose3 dictionary from frame ``b`` to frame ``a`` using an SE(3) transform."""
    return {i: aTb.compose(pose_b) for i, pose_b in pose_map_b.items()}


def Pose3s_with_sim3(poses_b: Sequence[Pose3], aSb: Similarity3) -> List[Pose3]:
    """Transport a list of Pose3s from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    return [aSb.transformFrom(pose_b) for pose_b in poses_b]


def optional_Pose3s_with_sim3(poses_b: Sequence[Optional[Pose3]], aSb: Similarity3) -> List[Optional[Pose3]]:
    """Transport an optional pose list from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    return [aSb.transformFrom(pose_b) if pose_b is not None else None for pose_b in poses_b]


def point_cloud_with_sim3(points_b: np.ndarray, aSb: Similarity3) -> np.ndarray:
    """Transport a point cloud from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    if points_b.size == 0:
        return points_b.copy()
    transformed_points = [np.asarray(aSb.transformFrom(point_b)) for point_b in points_b]
    return np.vstack(transformed_points)


def track_with_sim3(track_b: SfmTrack, aSb: Similarity3) -> SfmTrack:
    """Transport a single SfmTrack from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    track_a = SfmTrack(aSb.transformFrom(track_b.point3()))
    for k in range(track_b.numberMeasurements()):
        i, uv = track_b.measurement(k)
        track_a.addMeasurement(i, uv)
    return track_a


def tracks_with_sim3(tracks_b: Sequence[SfmTrack], aSb: Similarity3) -> List[SfmTrack]:
    """Transport a collection of SfmTracks from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    return [track_with_sim3(track_b, aSb) for track_b in tracks_b]


def camera_map_with_sim3(cameras_b: Mapping[int, CAMERA_TYPE], aSb: Similarity3) -> Dict[int, CAMERA_TYPE]:
    """Transport a camera dictionary from frame ``b`` to frame ``a`` using a Sim(3) transform."""
    cameras_a: Dict[int, CAMERA_TYPE] = {}
    for i, camera_b in cameras_b.items():
        if camera_b is None:
            continue
        new_pose = aSb.transformFrom(camera_b.pose())
        cameras_a[i] = create_camera(new_pose, camera_b.calibration())
    return cameras_a
