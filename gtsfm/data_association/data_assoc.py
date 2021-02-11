""" Create 2D-3D data association as a precursor to Bundle Adjustment.
1. Forms feature tracks from verified correspondences and global poses.
2. Estimates 3D landmark for each track (Ransac and simple triangulation modes available)
3. Filters tracks based on reprojection error.

References: 
1. Richard I. Hartley and Peter Sturm. Triangulation. Computer Vision and Image Understanding, Vol. 68, No. 2,
   November, pp. 146–157, 1997

Authors: Sushmita Warrier, Xiaolong Wu
"""
from typing import Dict, List, NamedTuple, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import PinholeCameraCal3Bundler, SfmData, SfmTrack

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.sfm_result import SfmResult
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationParam,
)


logger = logger_utils.get_logger()


class DataAssociation(NamedTuple):
    """Class to form feature tracks; for each track, call LandmarkInitializer.

    Args:
        reproj_error_thresh: the maximum reprojection error allowed.
        min_track_len: min length required for valid feature track / min nb of supporting views required for a landmark
                       to be valid.
        mode: triangulation mode, which dictates whether or not to use robust estimation.
        num_ransac_hypotheses (optional): number of hypothesis for RANSAC-based triangulation.
    """

    reproj_error_thresh: float
    min_track_len: int
    mode: TriangulationParam
    num_ransac_hypotheses: Optional[int] = None

    def __validate_track(self, sfm_track: Optional[SfmTrack]) -> bool:
        """Validate the track by checking its length."""
        return sfm_track is not None and sfm_track.number_measurements() >= self.min_track_len

    def run(
        self,
        cameras: Dict[int, PinholeCameraCal3Bundler],
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
    ) -> SfmData:
        """Perform the data association.

        Args:
            cameras: dictionary, with image index -> camera mapping.
            corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value as matching keypoint indices.
            keypoints_list: keypoints for each image.

        Returns:
            cameras and tracks as SfmData.
        """
        # generate tracks for 3D points using pairwise correspondences
        tracks = SfmTrack2d.generate_tracks_from_pairwise_matches(corr_idxs_dict, keypoints_list)

        # metrics on tracks w/o triangulation check
        num_tracks = len(tracks)
        track_lengths = list(map(lambda x: x.number_measurements(), tracks))

        logger.debug("[Data association] input number of tracks: %s", num_tracks)
        logger.debug(
            "[Data association] input avg. track length: %s",
            np.mean(track_lengths),
        )

        # initializer of 3D landmark for each track
        point3d_initializer = Point3dInitializer(
            cameras,
            self.mode,
            self.reproj_error_thresh,
            self.num_ransac_hypotheses,
        )

        # form SFMdata object after triangulation
        triangulated_data = SfmData()
        for track_2d in tracks:
            # triangulate and filter based on reprojection error
            sfm_track = point3d_initializer.triangulate(track_2d)

            if sfm_track is None:
                continue

            if self.__validate_track(sfm_track):
                triangulated_data.add_track(sfm_track)

        # TODO: improve dropped camera handling
        num_cameras = len(cameras.keys())
        expected_camera_indices = np.arange(num_cameras)
        # add cameras to landmark_map
        for i, cam in cameras.items():
            if i != expected_camera_indices[i]:
                raise RuntimeError("Some cameras must have been dropped ")
            triangulated_data.add_camera(cam)

        logger.debug(
            "[Data association] output number of tracks: %s",
            triangulated_data.number_tracks(),
        )
        logger.debug(
            "[Data association] output avg. track length: %s",
            SfmResult(triangulated_data, None).get_track_length_statistics()[0],
        )

        return triangulated_data

    def create_computation_graph(
        self,
        cameras: Delayed,
        corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        keypoints_graph: List[Delayed],
    ) -> Delayed:
        """Creates a computation graph for performing data association.

        Args:
            cameras: list of cameras wrapped up as Delayed.
            corr_idxs_graph: dictionary of correspondence indices, each value wrapped up as Delayed.
            keypoints_graph: list of wrapped up keypoints for each image.

        Returns:
            SfmData object wrapped up using dask.delayed.
        """
        return dask.delayed(self.run)(cameras, corr_idxs_graph, keypoints_graph)
