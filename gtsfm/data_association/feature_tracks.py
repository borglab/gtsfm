"""Utilities to generate and store tracks. Uses the Union-Find algorithm, with
image ID and keypoint index for that image as the unique keys.

A track is defined as a 2d measurement of a single 3d landmark seen in multiple
different images.

References:
1. P. Moulon, P. Monasse. Unordered Feature Tracking Made Fast and Easy, 2012, HAL Archives.
   https://hal-enpc.archives-ouvertes.fr/hal-00769267/file/moulon_monasse_featureTracking_CVMP12.pdf

Authors: Ayush Baid, Sushmita Warrier, John Lambert
"""
from typing import Dict, List, NamedTuple, Optional, Tuple

import gtsam
import numpy as np

from gtsfm.common.keypoints import Keypoints


class SfmMeasurement(NamedTuple):
    i: int  # camera index
    uv: np.ndarray  # 2d measurement

    def __eq__(self, other: object) -> bool:
        """Checks equality with the other object."""
        if not isinstance(other, SfmMeasurement):
            return False

        if self.i != other.i:
            return False

        return np.allclose(self.uv, other.uv)

    def __ne__(self, other: object) -> bool:
        """Checks inequality with the other object."""
        return not self == other


class SfmTrack(NamedTuple):
    measurements: List[SfmMeasurement]  # (cam_idx, 2D coordinates)
    landmark: Optional[np.ndarray] = None  # 3D landmark

    def number_measurements(self) -> int:
        """Returns the number of measurements."""
        return len(self.measurements)

    def measurement(self, idx: int) -> SfmMeasurement:
        """Getter for measurement at a particular index.

        Args:
            idx: index to fetch.

        Returns:
            measurement at the requested index.
        """
        return self.measurements[idx]

    def select_subset(self, idxs: List[int]) -> "SfmTrack":
        """Generates a new track with the subset of measurements.

        Returns:
            Track with the subset of measurements.
        """
        inlier_measurements = [self.measurements[j] for j in idxs]

        return SfmTrack(inlier_measurements, self.landmark)

    def __eq__(self, other: object) -> bool:
        """Checks equality with the other object."""

        # check object type
        if not isinstance(other, SfmTrack):
            return False

        # check number of measurements
        if len(self.measurements) != len(other.measurements):
            return False

        # check the individual measurements (order insensitive)
        # inefficient implementation but wont be used a lot
        for measurement in self.measurements:
            if measurement not in other.measurements:
                return False

        # finally, check the landmark
        if self.landmark is not None and other.landmark is None:
            return False
        elif self.landmark is None and other.landmark is not None:
            return False
        elif self.landmark is None and other.landmark is None:
            return True
        else:
            return np.allclose(self.landmark, other.landmark)

    def __ne__(self, other: object) -> bool:
        """Checks inequality with the other object."""
        return not self == other


def generate_tracks(
    matches_dict: Dict[Tuple[int, int], np.ndarray],
    keypoints_list: List[Keypoints],
) -> List[SfmTrack]:
    """Creates and filter tracks from matches.

    Creates a disjoint-set forest (DSF) and 2d tracks from pairwise matches. We create a
    singleton for union-find set elements from camera index of a detection and the index
    of that detection in that camera's keypoint list, i.e. (i,k).

    Args:
        matches_dict: Dict of pairwise matches of type:
                key: pose indices for the matched pair of images
                val: feature indices, as array of Nx2 shape; N being nb of features, and each
                    row is (feature_idx1, feature_idx2).
        keypoints_list: List of keypoints for each image.
    """

    # check to ensure dimensions of coordinates are correct
    dims_valid = all([kps.coordinates.ndim == 2 for kps in keypoints_list])
    if not dims_valid:
        raise Exception(
            "Dimensions for Keypoint coordinates incorrect. Array needs to be 2D"
        )

    # Generate the DSF to form tracks
    dsf = gtsam.DSFMapIndexPair()
    tracks_2d = []
    # for DSF finally
    # measurement_idxs represented by ks
    for (i1, i2), k_pairs in matches_dict.items():
        for (k1, k2) in k_pairs:
            dsf.merge(gtsam.IndexPair(i1, k1), gtsam.IndexPair(i2, k2))

    key_set = dsf.sets()
    # create a landmark map: a list of tracks
    # Each track is represented as a list of (camera_idx, measurements)
    for set_id in key_set:
        index_pair_set = key_set[
            set_id
        ]  # key_set is a wrapped C++ map, so this unusual syntax is required
        # Initialize track
        track_measurements = []
        for index_pair in gtsam.IndexPairSetAsArray(index_pair_set):
            # camera_idx is represented by i
            # measurement_idx is represented by k
            i = index_pair.i()
            k = index_pair.j()
            # add measurement in this track
            track_measurements += [
                SfmMeasurement(i, keypoints_list[i].coordinates[k])
            ]
        tracks_2d += [SfmTrack(track_measurements)]
    return delete_erroneous_tracks(tracks_2d)


def delete_erroneous_tracks(sfm_tracks_2d: List[SfmTrack]) -> List[SfmTrack]:
    """
    Delete tracks that have more than one measurement in the same image.

    Args:
        sfm_tracks_2d: feature tracks.

    Returns:
        filtered_tracks_2d: filtered feature tracks.
    """
    filtered_tracks_2d = []
    for sfm_track_2d in sfm_tracks_2d:
        track_cam_idxs = [
            measurement.i for measurement in sfm_track_2d.measurements
        ]
        if len(set(track_cam_idxs)) == len(track_cam_idxs):
            filtered_tracks_2d += [sfm_track_2d]

    return filtered_tracks_2d
