"""Utilities to generate and store tracks. Uses the Union-Find algorithm, with
image ID and keypoint index for that image as the unique keys.

A track is defined as a 2d measurement of a single 3d landmark seen in multiple
different images.

References:
1. P. Moulon, P. Monasse. Unordered Feature Tracking Made Fast and Easy, 2012, HAL Archives.
   https://hal-enpc.archives-ouvertes.fr/hal-00769267/file/moulon_monasse_featureTracking_CVMP12.pdf

Authors: Ayush Baid, Sushmita Warrier, John Lambert
"""
from typing import Dict, List, NamedTuple, Tuple

import gtsam
import numpy as np
from gtsam import Point2

from gtsfm.common.keypoints import Keypoints


class SfmMeasurement(NamedTuple):
    i: int  # camera index
    uv: np.ndarray  # 2d measurement


# equivalent to gtsam.SfmTrack, but without the 3d measurement
# (as we haven't triangulated it yet from 2d measurements)
class SfmTrack2d(NamedTuple):
    measurements: List[SfmMeasurement]

def generate_tracks(
    matches_dict: Dict[Tuple[int, int], np.ndarray],
    keypoints_list: List[Keypoints],
) -> List[SfmTrack2d]:
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
        tracks_2d += [SfmTrack2d(track_measurements)]
    return delete_erroneous_tracks(tracks_2d)


def delete_erroneous_tracks(
    sfm_tracks_2d: List[SfmTrack2d],
) -> List[SfmTrack2d]:
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
