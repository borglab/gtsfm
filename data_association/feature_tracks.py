"""Class to generate and store tracks. Uses the Union-Find algorithm, with the
image ID and keypoint index for that image as the unique keys.

A track is defined as a 2d measurement of a single 3d landmark seen in multiple different images.

Authors: Ayush Baid, Sushmita Warrier, John Lambert
"""
import gtsam
import numpy as np

from common.keypoints import Keypoints
from typing import Dict, List, NamedTuple, Tuple


class SfmMeasurement(NamedTuple):
    i: int  # camera index
    uv: np.ndarray  # 2d measurement


# equivalent to gtsam.SfmTrack, but without the 3d measurement
# (as we haven't triangulated it yet from 2d measurements)
class SfmTrack2d(NamedTuple):
    measurements: List[SfmMeasurement]


class FeatureTrackGenerator:
    """
    Creates and filter tracks from matches.
    """

    def __init__(
        self,
        matches_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
    ) -> None:
        """
        Creates DSF and landmark map from pairwise matches.

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
        self.filtered_landmark_data = []
        landmark_data = []
        # for DSF finally
        # measurement_idxs represented by ks
        for (i1, i2), k_pairs in matches_dict.items():
            for (k1, k2) in k_pairs:
                dsf.merge(gtsam.IndexPair(i1, k1), gtsam.IndexPair(i2, k2))

        key_set = dsf.sets()
        # create a landmark map: a list of tracks
        # Each track is represented as a list of (camera_idx, measurements)
        for s in key_set:
            key = key_set[
                s
            ]  # key_set is a wrapped C++ map, so this unusual syntax is required
            # Initialize track
            track = []
            for index_pair in gtsam.IndexPairSetAsArray(key):
                # camera_idx is represented by i
                # measurement_idx is represented by k
                i = index_pair.i()
                k = index_pair.j()
                # add measurement in this track
                track += [SfmMeasurement(i, keypoints_list[i].coordinates[k])]
            landmark_data += [SfmTrack2d(track)]
        self.filtered_landmark_data = self.delete_tracks(landmark_data)

    def delete_tracks(self, sfm_tracks_2d: List[SfmTrack2d]) -> List[SfmTrack2d]:
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
