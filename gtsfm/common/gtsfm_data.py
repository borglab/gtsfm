"""Class to hold the tracks and cameras of a 3D scene.

Authors: Ayush Baid
"""
import itertools
from typing import Dict, List, Optional

import numpy as np
from gtsam import PinholeCameraCal3Bundler, SfmTrack

import gtsfm.utils.graph as graph_utils

EQUALITY_TOLERANCE = 1e-5


class GtsfmData:
    """Class containing cameras and tracks, essentially describing the complete 3D scene.
    
    This class is needed over GTSAM's SfmData type because GTSAM's type does not allow for non-contiguous cameras.
    The situation of non-contiguous cameras can exists because of failures in front-end.
    """

    def __init__(self, number_images: int) -> None:
        """Initializes the class.

        Args:
            number_images: number of images/cameras in the scene.
        """
        self._cameras: Dict[int, PinholeCameraCal3Bundler] = {}
        self._tracks: List[SfmTrack] = []
        self._number_images = number_images

    def __eq__(self, other: object) -> bool:
        """Checks equality with the other object."""

        if not isinstance(other, GtsfmData):
            return False

        if self._number_images != other.number_images():
            return False

        for i, cam in self._cameras.items():
            other_cam = other.get_camera(i)
            if not cam.equals(other_cam, EQUALITY_TOLERANCE):
                return False

        for k in range(self.number_tracks()):
            track = self.get_track(k)
            other_track = other.get_track(k)

            if track.number_measurements() != other_track.number_measurements():
                return False

            for j in range(track.number_measurements()):
                i, uv = track.measurement(j)
                other_i, other_uv = other_track.measurement(j)

                if i != other_i:
                    return False
                if not np.allclose(uv, other_uv):
                    return False

        return True

    def number_images(self) -> int:
        """Getter for the number of images.

        Returns:
            Number of images.
        """
        return self._number_images

    def number_tracks(self) -> int:
        """Getter for the number of tracks.

        Returns:
            Number of tracks.
        """
        return len(self._tracks)

    def get_valid_camera_indices(self) -> List[int]:
        """Getter for image indices where there is a valid (not None) camera.

        Returns:
            List of indices with a valid camera.
        """
        return list(self._cameras.keys())

    def get_camera(self, index: int) -> Optional[PinholeCameraCal3Bundler]:
        """Getter for camera.

        Args:
            index: the image index to fetch the camera for.

        Returns:
            The camera if it is a valid one, None otherwise.
        """
        return self._cameras.get(index)

    def get_track(self, index: int) -> SfmTrack:
        """Getter for the track.

        Args:
            index: track index to fetch.

        Returns:
            Requested track.
        """
        return self._tracks[index]

    def add_track(self, track: SfmTrack) -> bool:
        """Add a track, after checking if all the cameras in the track are already added.

        Args:
            track: track to add.

        Returns:
            Flag indicating the success of adding operation.
        """
        # check if all cameras are already added
        for j in range(track.number_measurements()):
            i, _ = track.measurement(j)

            if i not in self._cameras:
                return False

        self._tracks.append(track)
        return True

    def add_camera(self, camera: PinholeCameraCal3Bundler, camera_index: int) -> None:
        """Adds a camera.

        Args:
            camera: camera object to it.
            camera_index: the index associated with this camera.

        Raises:
            ValueError: if the camera to be added is not a valid camera object.
        """
        if camera is None:
            raise ValueError("Camera cannot be None, should be a valid camera")
        self._cameras[camera_index] = camera

    def drop_cameras(self, camera_indices: List[int]) -> "GtsfmData":
        """Drops cameras and tracks associated with them.

        Returns:
            New object with the requested cameras (and associated tracks) dropped.
        """

        new_data = GtsfmData(self._number_images)

        for i, camera in self._cameras.items():
            if i not in camera_indices:
                new_data.add_camera(camera, i)

        new_camera_indices = new_data.get_valid_camera_indices()

        # add tracks which have all the camera present in new data
        for track in self._tracks:
            is_valid = True
            for m_idx in range(track.number_measurements()):
                i, _ = track.measurement(m_idx)
                if i not in new_camera_indices:
                    is_valid = False
                    break
            if is_valid:
                new_data.add_track(track)

        return new_data

    def select_largest_connected_component(self) -> "GtsfmData":
        """Selects the subset of data belonging to the largest connected component of the graph where the edges are
        between cameras which feature in the same track.

        Returns:
            New GtSfmData object with the subset of tracks and cameras.
        """
        camera_edges = []
        for sfm_track in self._tracks:
            cameras_in_use = []
            for m_idx in range(sfm_track.number_measurements()):
                i, _ = sfm_track.measurement(m_idx)
                cameras_in_use.append(i)

            # Recreate track connectivity from track information
            # For example: a track has cameras [0, 2, 5]. In that case we will add pairs (0, 2), (0, 5), (2, 5)
            camera_edges += list(itertools.combinations(cameras_in_use, 2))

        if len(camera_edges) == 0:
            return GtsfmData(self._number_images)

        cameras_in_largest_cc = graph_utils.get_nodes_in_largest_connected_component(camera_edges)

        # select cameras to drop and drop them
        dropped_cameras = np.setdiff1d(list(self._cameras.keys()), cameras_in_largest_cc)
        return self.drop_cameras(dropped_cameras)
