""" Placeholder for GTSFM data association module

Authors: Ayush Baid, John Lambert, Sushmita Warrier
"""
import random
from typing import Dict, List, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import PinholeCameraCal3Bundler, SfmData, SfmTrack

from common.keypoints import Keypoints


class DummyDataAssociation:
    def __init__(self, reproj_error_thresh: float, min_track_len: int):
        self.reproj_error_thresh = reproj_error_thresh
        self.min_track_len = min_track_len

    def run(
        self,
        cameras: Dict[int, PinholeCameraCal3Bundler],
        v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
    ) -> SfmData:
        """[summary]

        Args:
            cameras (Dict[int, PinholeCameraCal3Bundler]): [description]
            v_corr_idxs_dict (Dict[Tuple[int, int], np.ndarray]): [description]
            keypoints_list (List[Keypoints]): [description]

        Returns:
            cameras and tracks as SfmData
        """

        available_cams = np.array(list(cameras.keys()), dtype=np.uint32)

        # map the available cams index from 0....1

        # form few tracks randomly
        tracks = []
        num_tracks = random.randint(5, 10)

        for _ in range(num_tracks):
            track = []

            # randomly select cameras for this track
            selected_cams = np.random.choice(
                available_cams, self.min_track_len, replace=False
            )

            # obtain 3D point for the track
            point_3d = np.random.rand(3, 1)

            # create GTSAM's SfmTrack object
            sfmTrack = SfmTrack(point_3d)

            # for each selected camera, randomly select a point
            for cam_idx in selected_cams:
                measurement_idx = random.randint(0, len(keypoints_list[cam_idx]) - 1)
                measurement = keypoints_list[cam_idx].coordinates[measurement_idx]
                sfmTrack.add_measurement(cam_idx, measurement)

            tracks.append(sfmTrack)

        # create the final SfmData object
        sfmData = SfmData()
        for cam in cameras.values():
            sfmData.add_camera(cam)

        for track in tracks:
            sfmData.add_track(track)

        return SfmData

    def create_computation_graph(
        self,
        cameras: Delayed,
        v_corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        keypoints_graph: List[Delayed],
    ) -> Delayed:
        return dask.delayed(self.run)(cameras, v_corr_idxs_graph, keypoints_graph)
