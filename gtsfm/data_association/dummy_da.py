"""Placeholder for GTSFM data association module.

Authors: Ayush Baid, John Lambert, Sushmita Warrier
"""
import random
from typing import Dict, List, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import PinholeCameraCal3Bundler, SfmTrack

from gtsfm.common.keypoints import Keypoints
from gtsfm.common.gtsfm_data import GtsfmData


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
        num_images: int,
        cameras: Dict[int, PinholeCameraCal3Bundler],
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
    ) -> GtsfmData:
        """Perform the data association.

        Args:
            num_images: Number of images in the scene.
            cameras: dictionary, with image index -> camera mapping.
            corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value as matching keypoint indices.
            keypoints_list: keypoints for each image.

        Returns:
            Cameras and tracks as GtsfmData.
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
            selected_cams = np.random.choice(available_cams, self.min_track_len, replace=False)

            # for each selected camera, randomly select a point
            for cam_idx in selected_cams:
                measurement_idx = random.randint(0, len(keypoints_list[cam_idx]) - 1)
                measurement = keypoints_list[cam_idx].coordinates[measurement_idx]

                sfmTrack.add_measurement(cam_idx, measurement)

            tracks.append(sfmTrack)

        # create the final SfmData object
        gtsfm_data = GtsfmData(num_images)
        for i, cam in cameras.items():
            gtsfm_data.add_camera(cam, i)

        for track in tracks:
            gtsfm_data.add_track(track)

        return gtsfm_data

    def create_computation_graph(
        self,
        num_images: int,
        cameras: Delayed,
        corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        keypoints_graph: List[Delayed],
    ) -> Delayed:
        """Creates a computation graph for performing data association.

        Args:
            num_images: number of images in the scene.
            cameras: list of cameras wrapped up as Delayed.
            corr_idxs_graph: dictionary of correspondence indices, each value wrapped up as Delayed.
            keypoints_graph: list of wrapped up keypoints for each image.

        Returns:
            TODO: GtsfmData object wrapped up using dask.delayed.
        """
        return dask.delayed(self.run)(num_images, cameras, corr_idxs_graph, keypoints_graph)
