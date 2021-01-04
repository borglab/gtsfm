""" Placeholder for GTSFM data association module.

Authors: Ayush Baid, John Lambert, Sushmita Warrier
"""
import random
from typing import Dict, List, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import PinholeCameraCal3Bundler, SfmData, SfmTrack

from gtsfm.common.keypoints import Keypoints
from gtsfm.data_association.feature_tracks import SfmMeasurement


class DummyDataAssociation:
    """Class for generating random tracks and their 3D points for SfmData."""

    def __init__(self, reproj_error_thresh: float, min_track_len: int):
        """Initializes the hyperparameters.

        Args:
            reproj_error_thresh: the maximum reprojection error allowed.
            min_track_len: min length required for valid feature track.
        """
        self.reproj_error_thresh = reproj_error_thresh
        self.min_track_len = min_track_len

    def run(
        self,
        cameras: Dict[int, PinholeCameraCal3Bundler],
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
    ) -> SfmData:
        """Perform the data association.

        Args:
            cameras: dictionary with image index as key, and camera object w/
                     intrinsics + extrinsics as value.
            corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value
                            as matching keypoint indices.
            keypoints_list: keypoints for each image.

        Returns:
            cameras and tracks as SfmData
        """

        available_cams = np.array(list(cameras.keys()), dtype=np.uint32)

        # form few tracks randomly
        tracks = []
        num_tracks = random.randint(5, 10)

        for _ in range(num_tracks):
            # obtain 3D points for the track randomly
            point_3d = np.random.rand(3, 1)

            # create GTSAM's SfmTrack object
            sfmTrack = SfmTrack(point_3d)

            # randomly select cameras for this track
            selected_cams = np.random.choice(
                available_cams, self.min_track_len, replace=False
            )

            # for each selected camera, randomly select a point
            # measurements = []
            for cam_idx in selected_cams:
                measurement_idx = random.randint(
                    0, len(keypoints_list[cam_idx]) - 1
                )
                measurement = keypoints_list[cam_idx].coordinates[
                    measurement_idx
                ]

                # measurements.append(SfmMeasurement(cam_idx, measurement))

                sfmTrack.add_measurement(cam_idx, measurement)

            tracks.append(sfmTrack)

        # create the final SfmData object
        sfm_data = SfmData()
        for cam in cameras.values():
            sfm_data.add_camera(cam)

        for track in tracks:
            sfm_data.add_track(track)

        return sfm_data

    def create_computation_graph(
        self,
        cameras: Delayed,
        corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        keypoints_graph: List[Delayed],
    ) -> Delayed:
        """Creates a computation graph for performing data association.

        Args:
            cameras: list of cameras wrapped up as Delayed.
            corr_idxs_graph: dictionary of correspondence indices, each value
                             wrapped up as Delayed.
            keypoints_graph: list of wrapped up keypoints for each image.

        Returns:
            SfmData object wrapped up using dask.delayed.
        """
        return dask.delayed(self.run)(cameras, corr_idxs_graph, keypoints_graph)
