"""Estimates tracks from feature correspondences using the Union-Find algorithm.

The unique key for each measurement is a tuple of (Image ID, keypoint index for that image).

References:
1. P. Moulon, P. Monasse. Unordered Feature Tracking Made Fast and Easy, 2012, HAL Archives.
   https://hal-enpc.archives-ouvertes.fr/hal-00769267/file/moulon_monasse_featureTracking_CVMP12.pdf

Authors: Ayush Baid, Sushmita Warrier, John Lambert, Travis Driver
"""

import time
from typing import Dict, List, Tuple

import gtsam
import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.tracks_estimator_base import TracksEstimatorBase

logger = logger_utils.get_logger()


class CppDsfTracksEstimator(TracksEstimatorBase):
    """Estimates tracks using a disjoint-set forest (DSF), with core logic implemented in C++."""

    def run(self, matches_dict: Dict[Tuple[int, int], np.ndarray], keypoints_list: List[Keypoints]) -> List[SfmTrack2d]:
        """Estimate tracks from feature correspondences.

        Compare to GTSAM's `python/gtsam/tests/test_DsfTrackGenerator.py`.

        Creates a disjoint-set forest (DSF) and 2d tracks from pairwise matches. We create a singleton for union-find
        set elements from camera index of a detection and the index of that detection in that camera's keypoint list,
        i.e. (i,k).

        Args:
            matches_dict: Dict of pairwise matches of type:
                    key: indices for the matched pair of images
                    val: feature indices, as array of Nx2 shape; N being number of features. A row is (feature_idx1,
                         feature_idx2).
            keypoints_list: List of keypoints for each image.

        Returns:
            List of all valid SfmTrack2d generated by the matches.
        """
        start_time = time.time()
        # Check to ensure dimensions of coordinates are correct.
        dims_valid = all([kps.coordinates.ndim == 2 for kps in keypoints_list])
        if not dims_valid:
            raise Exception("Dimensions for Keypoint coordinates incorrect. Array needs to be 2D")

        # For each image pair (i1,i2), we provide a (K,2) matrix of corresponding keypoint indices (k1,k2).
        # (Converts python dict into gtsam.MatchIndicesMap.)
        matches_map = gtsam.MatchIndicesMap()
        for (i1, i2), corr_idxs in matches_dict.items():
            matches_map[gtsam.IndexPair(i1, i2)] = corr_idxs

        # Convert gtsfm Keypoints into gtsam Keypoints.
        keypoints_vector = gtsam.KeypointsVector()
        for keypoint in keypoints_list:
            keypoints_vector.append(gtsam.gtsfm.Keypoints(keypoint.coordinates))

        tracks = gtsam.gtsfm.tracksFromPairwiseMatches(
            matches_map,
            keypoints_vector,
            verbose=True,
        )

        track_2d_list = []
        for track in tracks:
            # Converting gtsam SfmTrack2d into gtsfm SfmTrack2d.
            # Note: `indexVector` contains the camera indices for each of the (K,) 2d (u,v) measurements.
            # `measurementMatrix` contains the measurements as a 2D matrix (K,2).
            track_ = SfmTrack2d(
                [SfmMeasurement(i, uv) for (i, uv) in zip(track.indexVector(), track.measurementMatrix())]
            )
            track_2d_list.append(track_)

        duration = time.time() - start_time
        logger.info("CppDsfTracksEstimator took %.2f sec. to estimate %d tracks.", duration, track_2d_list)
        return track_2d_list
