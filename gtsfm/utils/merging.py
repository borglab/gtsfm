"""Utility functions for merging geometry defined in different coordinate frames.

Authors: Richi Dubey, Frank Dellaert
"""

from __future__ import annotations

from typing import Dict, Mapping

from gtsam import Pose3, Similarity3  # type: ignore

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import transform as transform_utils


def merge_pose_maps(a: Mapping[int, Pose3], b: Mapping[int, Pose3], aTb: Pose3) -> Dict[int, Pose3]:
    """Merge two pose dictionaries without mutating the inputs.

    Args:
        a: Pose dictionary expressed in frame ``a``.
        b: Pose dictionary expressed in frame ``b``.
        aTb: Transform taking a pose from frame ``b`` into frame ``a``.

    Returns:
        Dictionary containing poses from ``a`` and poses from ``b`` expressed in frame ``a``.
    """
    merged: Dict[int, Pose3] = dict(a)
    transformed_b = transform_utils.transform_pose_map(b, aTb)
    for key, pose in transformed_b.items():
        if key not in merged:
            merged[key] = pose
    return merged


def merge(a: GtsfmData, b: GtsfmData, aSb: Similarity3) -> GtsfmData:
    """Merge two ``GtsfmData`` objects into a new ``GtsfmData`` expressed in frame ``a``.

    Args:
        a: Scene expressed in frame ``a``.
        b: Scene expressed in frame ``b``.
        aSb: Transform taking geometry from frame ``b`` into frame ``a``.

    Returns:
        New ``GtsfmData`` instance containing geometry from both inputs.
    """
    merged_cameras = dict(a.cameras())
    transformed_b_cameras = transform_utils.transform_camera_map(b.cameras(), aSb)
    for key, camera in transformed_b_cameras.items():
        if key not in merged_cameras:
            merged_cameras[key] = camera

    merged_tracks = list(a.tracks())
    merged_tracks.extend(transform_utils.transform_tracks(b.tracks(), aSb))

    merged_data = GtsfmData(number_images=max(a.number_images(), b.number_images()))
    for key, camera in merged_cameras.items():
        merged_data.add_camera(key, camera)
    for track in merged_tracks:
        if not merged_data.add_track(track):
            # Preserve tracks even if some measurements reference cameras missing from the dataset.
            merged_data._tracks.append(track)

    return merged_data
