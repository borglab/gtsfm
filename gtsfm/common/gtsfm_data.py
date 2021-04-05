"""Class to hold the tracks and cameras of a 3D scene.
This can be the output of either data association or of bundle adjustment.

Authors: Ayush Baid, John Lambert, Xiaolong Wu
"""
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from gtsam import PinholeCameraCal3Bundler, Pose3, SfmTrack

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.reprojection as reproj_utils

logger = logger_utils.get_logger()

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

        for j in range(self.number_tracks()):
            track = self.get_track(j)
            other_track = other.get_track(j)

            if track.number_measurements() != other_track.number_measurements():
                return False

            for k in range(track.number_measurements()):
                i, uv = track.measurement(k)
                other_i, other_uv = other_track.measurement(k)

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

    def get_camera_poses(self) -> List[Optional[Pose3]]:
        """Getter for camera poses wTi.

        This function returns the pose for all cameras (equal to number_images in GtsfmData), even if they were not
        computed by the pipeline.

        Returns:
            camera poses as a list, each representing wTi
        """
        cameras = [self.get_camera(i) for i in range(self.number_images())]
        poses = [camera.pose() if camera is not None else None for camera in cameras]

        return poses

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

    def add_camera(self, index: int, camera: PinholeCameraCal3Bundler) -> None:
        """Adds a camera.

        Args:
            index: the index associated with this camera.
            camera: camera object to it.

        Raises:
            ValueError: if the camera to be added is not a valid camera object.
        """
        if camera is None:
            raise ValueError("Camera cannot be None, should be a valid camera")
        self._cameras[index] = camera

    def get_track_length_statistics(self) -> Tuple[float, float]:
        """Compute mean and median lengths of all the tracks.

        Returns:
            Mean track length.
            Median track length.
        """
        if self.number_tracks() == 0:
            return 0, 0

        track_lengths = self.get_track_lengths()
        return np.mean(track_lengths), np.median(track_lengths)

    def get_track_lengths(self) -> np.ndarray:
        """Get an array containing the lengths of all tracks.

        Returns:
            Array containing all track lengths.
        """
        if self.number_tracks() == 0:
            return np.array([], dtype=np.uint32)

        track_lengths = [
            self.get_track(j).number_measurements() for j in range(self.number_tracks())
        ]
        return np.array(track_lengths, dtype=np.uint32)

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
        logger.info(
            "Largest connected component contains {} of {} cameras returned by front-end (of {} total imgs)".format(
                len(cameras_in_largest_cc), len(self.get_valid_camera_indices()), self._number_images
            )
        )
        return GtsfmData.from_selected_cameras(self, cameras_in_largest_cc)

    @classmethod
    def from_selected_cameras(cls, gtsfm_data: "GtsfmData", camera_indices: List[int]) -> "GtsfmData":
        """Selects the cameras in the input list and the tracks associated with those cameras.

        Args:
            gtsfm_data: data to pick the cameras from.
            camera_indices: camera indices to select and keep in the new data.

        Returns:
            New object with the selected cameras and associated tracks.
        """
        new_data = cls(gtsfm_data.number_images())

        for i in gtsfm_data.get_valid_camera_indices():
            if i in camera_indices:
                new_data.add_camera(i, gtsfm_data.get_camera(i))

        new_camera_indices = new_data.get_valid_camera_indices()

        # add tracks which have all the camera present in new data
        for j in range(gtsfm_data.number_tracks()):
            track = gtsfm_data.get_track(j)
            is_valid = True
            for k in range(track.number_measurements()):
                i, _ = track.measurement(k)
                if i not in new_camera_indices:
                    is_valid = False
                    break
            if is_valid:
                new_data.add_track(track)

        return new_data

    def get_scene_avg_reprojection_error(self) -> float:
        """Get average reprojection error for all 3d points in the entire scene

        Returns:
            scene_avg_reproj_error: average of reprojection errors for every 3d point to its 2d measurements
        """
        scene_reproj_errors = []
        for track in self._tracks:
            track_errors, _ = reproj_utils.compute_track_reprojection_errors(self._cameras, track)
            scene_reproj_errors.extend(track_errors)

        scene_avg_repoj_error = np.mean(scene_reproj_errors)

        scene_reproj_errors = np.array(scene_reproj_errors)
        logger.info("Min scene reproj error: {}".format(scene_reproj_errors.min()))
        logger.info("Avg scene reproj error: {}".format(scene_reproj_errors.mean()))
        logger.info("Median scene reproj error: {}".format(np.median(scene_reproj_errors)))
        logger.info("Max scene reproj error: {}".format(scene_reproj_errors.max()))
        return scene_avg_repoj_error

    def __validate_track(self, track: SfmTrack, reproj_err_thresh: float) -> bool:
        """Validates a track based on reprojection errors and cheirality checks.

        Args:
            track: track with 3D landmark and measurements.
            reproj_err_thresh: reprojection err threshold for each measurement.

        Returns:
            validity of the track.
        """
        errors, avg_reproj_error = reproj_utils.compute_track_reprojection_errors(self._cameras, track)
        # track is valid as all measurements have error below the threshold
        cheirality_success = np.all(~np.isnan(errors))
        return np.all(errors < reproj_err_thresh) and cheirality_success

    def filter_landmarks(self, reproj_err_thresh: float = 5) -> "GtsfmData":
        """Filters out landmarks with high reprojection error

        Args:
            reproj_err_thresh: reprojection err threshold for each measurement.
        """
        # TODO: move this function to utils or GTSAM
        filtered_data = GtsfmData(self.number_images())

        # add all the cameras
        for i in self.get_valid_camera_indices():
            filtered_data.add_camera(i, self.get_camera(i))

        for j in range(self.number_tracks()):
            track = self.get_track(j)

            if self.__validate_track(track, reproj_err_thresh):
                filtered_data.add_track(track)

        return filtered_data

