"""Class to hold the tracks and cameras of a 3D scene.
This can be the output of either data association or of bundle adjustment.

Authors: Ayush Baid, John Lambert, Xiaolong Wu
"""

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import gtsam  # type: ignore
import numpy as np
from gtsam import Pose3, SfmTrack, Similarity3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.graph as graph_utils
import gtsfm.utils.images as image_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.reprojection as reprojection
from gtsfm.utils import transform as transform_utils
import thirdparty.colmap.scripts.python.read_write_model as colmap_io
from gtsfm.common.image import Image
from gtsfm.evaluation.metrics import GtsfmMetric
from gtsfm.products.visibility_graph import ImageIndexPairs
from gtsfm.utils.alignment import estimate_sim3_ignore_missing
from gtsfm.utils.pycolmap_utils import gtsfm_calibration_to_colmap_camera

logger = logger_utils.get_logger()

EQUALITY_TOLERANCE = 1e-5


class GtsfmData:
    """Class containing cameras and tracks, essentially describing the complete 3D scene.

    This class is needed over GTSAM's SfmData type because GTSAM's type does not allow for non-contiguous cameras.
    The situation of non-contiguous cameras can exists because of failures in front-end.
    """

    def __init__(
        self,
        number_images: int,
        cameras: Optional[Mapping[int, gtsfm_types.CAMERA_TYPE]] = None,
        tracks: Optional[List[SfmTrack]] = None,
    ) -> None:
        """Initializes the class.

        Args:
            number_images: number of images/cameras in the scene.
            cameras: Cameras in scene.
            tracks: SfmTracks in scene.
        """
        self._number_images = number_images
        self._cameras: dict[int, gtsfm_types.CAMERA_TYPE] = {}
        self._tracks: List[SfmTrack] = []

        # Initialize from inputs if provided.
        if cameras is not None:
            for i, cam in cameras.items():
                self.add_camera(i, cam)
        if tracks is not None:
            for track in tracks:
                self.add_track(track)

    def __repr__(self) -> str:
        """String representation of the object."""
        return (
            f"GtsfmData("
            f"num_images={self._number_images}, "
            f"num_cameras={len(self._cameras)}, "
            f"num_tracks={len(self._tracks)})"
        )

    @classmethod
    def from_sfm_data(cls, sfm_data: gtsam.SfmData) -> "GtsfmData":
        """Initialize from gtsam.SfmData instance.

        Args:
            sfm_data: camera parameters and point tracks.

        Returns:
            A new GtsfmData instance.
        """
        num_images = sfm_data.numberCameras()
        gtsfm_data = cls(num_images)
        for i in range(num_images):
            camera = sfm_data.camera(i)
            gtsfm_data.add_camera(i, camera)
        for j in range(sfm_data.numberTracks()):
            gtsfm_data.add_track(sfm_data.track(j))

        return gtsfm_data

    @classmethod
    def read_colmap(cls, data_dir: str) -> "GtsfmData":
        """Reads in full scene reconstruction model from scene data stored in the COLMAP file format.

        Reference: https://colmap.github.io/format.html

        Args:
            data_dir: This directory should contain 3 files: either `cameras.txt`, `images.txt`, and `points3D.txt`, or
                `cameras.bin`, `images.bin`, and `points3D.bin`.

        Returns:
            A new GtsfmData instance.
        """
        # Determine whether scene data is stored in a text (txt) or binary (bin) file format.
        if Path(data_dir, "images.txt").exists():
            file_format = ".txt"
        elif Path(data_dir, "images.bin").exists():
            file_format = ".bin"
        else:
            raise ValueError(
                f"Unknown file format, as neither `{data_dir}/images.txt` or `{data_dir}/images.bin` could be found."
            )
        cameras, images, points3d = colmap_io.read_model(path=data_dir, ext=file_format)
        tracks = io_utils.tracks_from_colmap(images, points3d)
        image_data = io_utils.image_data_from_colmap(cameras, images)
        gtsam_cameras = {}
        for i, (_, wTi, calibration, _) in enumerate(image_data):
            camera_type = gtsfm_types.get_camera_class_for_calibration(calibration)
            gtsam_cameras[i] = camera_type(wTi, calibration)  # type: ignore

        return cls(len(images), gtsam_cameras, tracks)

    @classmethod
    def read_bal(cls, file_path: str) -> "GtsfmData":
        """Read a Bundle Adjustment in the Large" (BAL) file.

        See https://grail.cs.washington.edu/projects/bal/ for more details on the format.

        Args:
            file_path: File path of the BAL file.

        Returns:
            The data as a GtsfmData object.
        """
        sfm_data = gtsam.readBal(file_path)
        return cls.from_sfm_data(sfm_data)

    @classmethod
    def read_bundler(cls, file_path: str) -> "GtsfmData":
        """Read a Bundler file.

        Args:
            file_path: File path of the Bundler file.

        Returns:
            The data as a GtsfmData object.
        """
        sfm_data = gtsam.SfmData.FromBundlerFile(file_path)
        return cls.from_sfm_data(sfm_data)

    def __eq__(self, other: object) -> bool:
        """Checks equality with the other object."""

        if not isinstance(other, GtsfmData):
            return False

        if self._number_images != other.number_images():
            return False

        for i, cam in self._cameras.items():
            other_cam = other.get_camera(i)
            assert other_cam is not None
            if not cam.equals(other_cam, EQUALITY_TOLERANCE):  # type: ignore
                return False

        for j in range(self.number_tracks()):
            track = self.get_track(j)
            other_track = other.get_track(j)

            if track.numberMeasurements() != other_track.numberMeasurements():
                return False

            for k in range(track.numberMeasurements()):
                i, uv = track.measurement(k)
                other_i, other_uv = other_track.measurement(k)

                if i != other_i:
                    return False
                if not np.allclose(uv, other_uv):
                    return False

        return True

    def number_images(self) -> int:
        """Returns the number of images."""
        return self._number_images

    def tracks(self) -> List[SfmTrack]:
        """Returns all tracks."""
        return self._tracks

    def number_tracks(self) -> int:
        """Returns the number of tracks."""
        return len(self._tracks)

    def get_valid_camera_indices(self) -> List[int]:
        """Returns indices of valid cameras."""
        return list(self._cameras.keys())

    def cameras(self) -> Dict[int, gtsfm_types.CAMERA_TYPE]:
        """Returns a dictionary of all cameras indexed by their image indices."""
        return self._cameras

    def get_camera(self, index: int) -> Optional[gtsfm_types.CAMERA_TYPE]:
        """Returns camera for given index, or None."""
        return self._cameras.get(index)

    def poses(self) -> Dict[int, Pose3]:
        """Returns poses as a dictionary, without missing poses."""
        return {i: cam.pose() for i, cam in self._cameras.items() if cam is not None}

    def get_camera_poses(self) -> List[Optional[Pose3]]:
        """Returns poses for all cameras (wTi), including missing ones as None."""
        cameras = [self.get_camera(i) for i in range(self.number_images())]
        poses = [camera.pose() if camera is not None else None for camera in cameras]
        return poses

    def get_track(self, index: int) -> SfmTrack:
        """Returns track at given index."""
        return self._tracks[index]

    def add_track(self, track: SfmTrack) -> bool:
        """Adds a track if all cameras exist; returns success flag."""
        for j in range(track.numberMeasurements()):
            i, _ = track.measurement(j)
            if i not in self._cameras:
                return False
        self._tracks.append(track)
        return True

    def get_tracks(self) -> List[SfmTrack]:
        """Returns all tracks."""
        return self._tracks

    def add_camera(self, index: int, camera: gtsfm_types.CAMERA_TYPE) -> None:
        """Adds camera at index if not already present."""
        if camera is None:
            raise ValueError("Camera cannot be None, should be a valid camera")
        if index not in self._cameras:
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
        return float(np.mean(track_lengths)), float(np.median(track_lengths))

    def get_track_lengths(self) -> np.ndarray:
        """Get an array containing the lengths of all tracks.

        Returns:
            Array containing all track lengths.
        """
        if self.number_tracks() == 0:
            return np.array([], dtype=np.uint32)

        track_lengths = [self.get_track(j).numberMeasurements() for j in range(self.number_tracks())]
        return np.array(track_lengths, dtype=np.uint32)

    def select_largest_connected_component(self, extra_camera_edges: Optional[ImageIndexPairs] = None) -> "GtsfmData":
        """Selects the subset of data belonging to the largest connected component of the graph where the edges are
        between cameras which feature in the same track.

        Args:
            extra_camera_edges (optional): edges which are to be considered as part of the graph when computing
                                           connected components.
        Returns:
            New GtSfmData object with the subset of tracks and cameras.
        """
        camera_edges = []
        for sfm_track in self._tracks:
            cameras_in_use = []
            for m_idx in range(sfm_track.numberMeasurements()):
                i, _ = sfm_track.measurement(m_idx)
                cameras_in_use.append(i)

            # Recreate track connectivity from track information
            # For example: a track has cameras [0, 2, 5]. In that case we will add pairs (0, 2), (0, 5), (2, 5)
            camera_edges += list(itertools.combinations(cameras_in_use, 2))

        # TODO(Ayush): add unit tests for extra camera edges
        if extra_camera_edges is not None:
            camera_edges += extra_camera_edges

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
    def from_cameras_and_tracks(
        cls, cameras: Mapping[int, gtsfm_types.CAMERA_TYPE], tracks: List[SfmTrack], number_images: int
    ) -> "GtsfmData":
        """Creates a GtsfmData object from a pre-existing set of cameras and tracks."""
        new_data = cls(number_images=number_images)
        new_data._cameras = dict(cameras)
        new_data._tracks = tracks
        return new_data

    @classmethod
    def from_selected_cameras(cls, gtsfm_data: "GtsfmData", camera_indices: List[int]) -> "GtsfmData":
        """Selects the cameras in the input list and the tracks associated with those cameras.

        Args:
            gtsfm_data: data to pick the cameras from.
            camera_indices: camera indices to select and keep in the new data.

        Returns:
            New object with the selected cameras and associated tracks.
        """
        new_data = cls(number_images=gtsfm_data.number_images())

        for i in gtsfm_data.get_valid_camera_indices():
            if i in camera_indices:
                camera_i = gtsfm_data.get_camera(i)
                assert camera_i is not None
                new_data.add_camera(i, camera_i)

        new_camera_indices = new_data.get_valid_camera_indices()

        # add tracks which have all the camera present in new data
        for j in range(gtsfm_data.number_tracks()):
            track = gtsfm_data.get_track(j)
            is_valid = True
            for k in range(track.numberMeasurements()):
                i, _ = track.measurement(k)
                if i not in new_camera_indices:
                    is_valid = False
                    break
            if is_valid:
                new_data.add_track(track)

        return new_data

    def get_scene_reprojection_errors(self) -> np.ndarray:
        """Get the scene reprojection errors for all 3D points and all associated measurements.

        Returns:
            Reprojection errors (measured in pixels) as a 1D numpy array.
        """
        scene_reproj_errors: List[float] = []
        for track in self._tracks:
            track_errors, _ = reprojection.compute_track_reprojection_errors(self._cameras, track)
            # passing an array argument to .extend() will convert the array to a list, and append its elements
            scene_reproj_errors.extend(track_errors)

        return np.array(scene_reproj_errors)

    def aggregate_metrics(self) -> Mapping[str, Any]:
        """Aggregate metrics about the reprojection errors and 3d track lengths (summary stats).

        Args:
            ba_data: bundle adjustment result

        Returns:
            dictionary containing metrics of bundle adjustment result
        """
        track_lengths_3d = self.get_track_lengths()
        scene_reproj_errors = self.get_scene_reprojection_errors()

        def convert_to_rounded_float(x):
            return int(np.round(x, 3))

        stats_dict: dict[str, int | dict[str, int]] = {}
        stats_dict["number_tracks"] = self.number_tracks()
        stats_dict["3d_track_lengths"] = {
            "min": convert_to_rounded_float(track_lengths_3d.min()),
            "mean": convert_to_rounded_float(np.mean(track_lengths_3d)),
            "median": convert_to_rounded_float(np.median(track_lengths_3d)),
            "max": convert_to_rounded_float(track_lengths_3d.max()),
        }
        stats_dict["reprojection_errors_px"] = {
            "min": convert_to_rounded_float(np.nanmin(scene_reproj_errors)),
            "mean": convert_to_rounded_float(np.nanmean(scene_reproj_errors)),
            "median": convert_to_rounded_float(np.nanmedian(scene_reproj_errors)),
            "max": convert_to_rounded_float(np.nanmax(scene_reproj_errors)),
        }
        return stats_dict

    def get_avg_scene_reprojection_error(self) -> float:
        """Get average reprojection error for all 3d points in the entire scene

        Returns:
            Average of reprojection errors for every 3d point to its 2d measurements
        """
        scene_reproj_errors = self.get_scene_reprojection_errors()
        scene_avg_reproj_error = np.nan if np.isnan(scene_reproj_errors).all() else np.nanmean(scene_reproj_errors)
        return float(scene_avg_reproj_error)

    def log_scene_reprojection_error_stats(self) -> None:
        """Logs reprojection error stats for all 3d points in the entire scene."""
        scene_reproj_errors = self.get_scene_reprojection_errors()
        logger.info(
            "Min scene reproj error: %.3f", np.nanmin(scene_reproj_errors) if len(scene_reproj_errors) else np.nan
        )
        logger.info(
            "Avg scene reproj error: %.3f", np.nanmean(scene_reproj_errors) if len(scene_reproj_errors) else np.nan
        )
        logger.info(
            "Median scene reproj error: %.3f", np.nanmedian(scene_reproj_errors) if len(scene_reproj_errors) else np.nan
        )
        logger.info(
            "Max scene reproj error: %.3f", np.nanmax(scene_reproj_errors) if len(scene_reproj_errors) else np.nan
        )

    def __validate_track(self, track: SfmTrack, reproj_err_thresh: float) -> bool:
        """Validates a track based on reprojection errors and cheirality checks.

        Args:
            track: track with 3D landmark and measurements.
            reproj_err_thresh: reprojection err threshold for each measurement.

        Returns:
            validity of the track.
        """
        errors, avg_reproj_error = reprojection.compute_track_reprojection_errors(self._cameras, track)
        # track is valid as all measurements have error below the threshold
        cheirality_success = np.all(~np.isnan(errors))
        return bool(np.all(errors < reproj_err_thresh) and cheirality_success)

    def filter_landmarks(self, reproj_err_thresh: float = 5) -> Tuple["GtsfmData", List[bool]]:
        """Filters out landmarks with high reprojection error

        Args:
            reproj_err_thresh: reprojection err threshold for each measurement.

        Returns:
            New instance, and list of valid flags, one for each track.
        """
        # TODO: move this function to utils or GTSAM
        filtered_data = GtsfmData(self.number_images())

        valid_mask = [self.__validate_track(track, reproj_err_thresh) for track in self._tracks]

        for track, valid in zip(self._tracks, valid_mask):
            if not valid:
                continue
            # check if all cameras with measurement in this track have already been added
            for k in range(track.numberMeasurements()):
                i, _ = track.measurement(k)
                camera_i = self.get_camera(i)
                assert camera_i is not None
                filtered_data.add_camera(i, camera_i)
            filtered_data.add_track(track)

        return filtered_data, valid_mask

    def aligned_to_poses_via_sim3(self, aTi_list: Sequence[Optional[Pose3]]) -> "GtsfmData":
        """Return a copy of the scene aligned to the supplied reference poses via Sim(3).

        Args:
            wTi_list_ref: Reference/target camera poses, ordered by camera index.

        Returns:
            New GtsfmData aligned to the reference pose graph.
        """
        bTi_list = self.get_camera_poses()
        aSb = estimate_sim3_ignore_missing(aTi_list, bTi_list)
        return self.apply_Sim3(aSb)

    def get_metrics(self, suffix: str, store_full_data: bool = False) -> List[GtsfmMetric]:
        """Helper to get bundle adjustment metrics from a GtsfmData object with a suffix for metric names."""
        metrics = []
        metrics.append(GtsfmMetric(name="number_cameras", data=len(self.get_valid_camera_indices())))
        metrics.append(GtsfmMetric("number_tracks" + suffix, self.number_tracks()))
        metrics.append(
            GtsfmMetric(
                name="3d_track_lengths" + suffix,
                data=self.get_track_lengths(),
                plot_type=GtsfmMetric.PlotType.HISTOGRAM,
                store_full_data=store_full_data,
            )
        )
        metrics.append(
            GtsfmMetric(
                name=f"reprojection_errors{suffix}_px",
                data=self.get_scene_reprojection_errors(),
                store_full_data=store_full_data,
                plot_type=GtsfmMetric.PlotType.BOX,
            )
        )
        return metrics

    def apply_Sim3(self, aSb: Similarity3) -> "GtsfmData":
        """Assume current tracks and cameras are in frame "b", then transport them to frame "a".

        Returns:
            New GtsfmData object which has been transformed from frame a to frame b.
        """
        bTi_list = self.get_camera_poses()
        aTi_list = [aSb.transformFrom(bTi) if bTi is not None else None for bTi in bTi_list]
        aligned_data = GtsfmData(number_images=self.number_images())

        # Update the camera poses to their aligned poses, but use the previous calibration.
        for i, aTi in enumerate(aTi_list):
            if aTi is None:
                continue
            camera_i = self.get_camera(i)
            assert camera_i is not None
            calibration = camera_i.calibration()
            camera_type = gtsfm_types.get_camera_class_for_calibration(calibration)
            aligned_data.add_camera(i, camera_type(aTi, calibration))  # type: ignore
        # Align estimated tracks to ground truth.
        for j in range(self.number_tracks()):
            # Align each 3d point
            track_b = self.get_track(index=j)
            # Place into the "a" reference frame
            pt_a = aSb.transformFrom(track_b.point3())
            track_a = SfmTrack(pt_a)
            # Copy over the 2d measurements directly into the new track.
            for k in range(track_b.numberMeasurements()):
                i, uv = track_b.measurement(k)
                track_a.addMeasurement(i, uv)
            aligned_data.add_track(track_a)

        return aligned_data

    def merged_with(self, other: "GtsfmData", aSb: Similarity3) -> "GtsfmData":
        """Return a new scene containing self and ``other`` expressed in this scene's frame.

        Args:
            other: Scene to merge, currently expressed in frame ``b``.
            aSb: Transform taking geometry from frame ``b`` to frame ``a`` (this scene).

        Returns:
            New ``GtsfmData`` containing cameras and tracks from both inputs.
        """
        merged_cameras = dict(self.cameras())
        transformed_other_cameras = transform_utils.transform_camera_map(other.cameras(), aSb)
        for key, camera in transformed_other_cameras.items():
            if key not in merged_cameras:
                merged_cameras[key] = camera

        merged_tracks = list(self.tracks())
        merged_tracks.extend(transform_utils.transform_tracks(other.tracks(), aSb))

        max_camera_index = max(merged_cameras.keys()) if merged_cameras else -1
        number_images = max(self.number_images(), other.number_images(), max_camera_index + 1)
        merged_data = GtsfmData(number_images=number_images)

        for key, camera in merged_cameras.items():
            merged_data.add_camera(key, camera)

        for track in merged_tracks:
            if not merged_data.add_track(track):
                merged_data._tracks.append(track)

        return merged_data

    def downsample(self, fraction_points_to_keep: float, seed: int = 42) -> "GtsfmData":
        """Downsample the number of 3D points in the scene by randomly selecting a fraction of them."""
        # TODO(Frank): extend to downsample cameras as well
        rng = np.random.default_rng(seed)
        num_tracks = self.number_tracks()
        indices_to_keep = rng.choice(num_tracks, size=int(num_tracks * fraction_points_to_keep), replace=False)
        downsampled_tracks = [self._tracks[idx] for idx in indices_to_keep]
        return GtsfmData(self.number_images(), self.cameras(), downsampled_tracks)

    # COLMAP export functions

    def write_cameras(self, save_dir: str | Path, image_shapes: List[Tuple[int, ...]]) -> None:
        """Writes the camera data file in the COLMAP format.

        Reference: https://colmap.github.io/format.html#cameras-txt

        Args:
            save_dir: Folder to put the cameras.txt file in.
            image_shapes: List of all image shapes for this scene, in order of image index.
        """
        dir_path = Path(save_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        # TODO: handle shared intrinsics
        file_path = dir_path / "cameras.txt"
        with open(file_path, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            # note that we save the number of estimated cameras, not the number of input images,
            # which would instead be self.number_images().
            f.write(f"# Number of cameras: {len(self.get_valid_camera_indices())}\n")

            for i in self.get_valid_camera_indices():
                camera_i = self.get_camera(i)
                assert camera_i is not None, "Camera %d is None" % i
                gtsfm_cal = camera_i.calibration()
                shape_i = image_shapes[i]
                colmap_cam = gtsfm_calibration_to_colmap_camera(i, gtsfm_cal, shape_i[0], shape_i[1])
                to_write = [colmap_cam.id, colmap_cam.model, colmap_cam.width, colmap_cam.height, *colmap_cam.params]
                line = " ".join([str(elem) for elem in to_write])
                f.write(line + "\n")

    def write_images(self, save_dir: str | Path, image_filenames: Sequence[str | None]) -> None:
        """Writes the image data file in the COLMAP format.

        Reference: https://colmap.github.io/format.html#images-txt
        Note: the "Number of images" saved to the .txt file is not the number of images
        fed to the SfM algorithm, but rather the number of localized camera poses/images,
        which COLMAP refers to as the "reconstructed cameras".

        Args:
            image_filenames: List of all image file names for this scene, in order of image index.
            save_dir: Folder to put the images.txt file in.
        """
        dir_path = Path(save_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        num_imgs = self.number_images()

        image_id_num_measurements: defaultdict[int, int] = defaultdict(int)
        for j in range(self.number_tracks()):
            track = self.get_track(j)
            for k in range(track.numberMeasurements()):
                image_id, uv_measured = track.measurement(k)
                image_id_num_measurements[image_id] += 1
        mean_obs_per_img = (
            sum(image_id_num_measurements.values()) / len(image_id_num_measurements)
            if len(image_id_num_measurements)
            else 0
        )

        file_path = dir_path / "images.txt"
        with open(file_path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {num_imgs}, mean observations per image: {mean_obs_per_img:.3f}\n")

            for i in self.get_valid_camera_indices():
                camera = self.get_camera(i)
                assert camera is not None, "Camera %d is None" % i
                # COLMAP exports camera extrinsics (cTw), not the poses (wTc), so must invert
                iTw = camera.pose().inverse()
                iRw_quaternion = iTw.rotation().toQuaternion()
                itw = iTw.translation()
                tx, ty, tz = itw
                qw, qx, qy, qz = iRw_quaternion.w(), iRw_quaternion.x(), iRw_quaternion.y(), iRw_quaternion.z()

                f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {image_filenames[i]}\n")

                # write out points2d
                for j in range(self.number_tracks()):
                    track = self.get_track(j)
                    for k in range(track.numberMeasurements()):
                        # write each measurement
                        image_id, uv_measured = track.measurement(k)
                        if image_id == i:
                            f.write(f" {uv_measured[0]:.3f} {uv_measured[1]:.3f} {j}")
                f.write("\n")

    def write_points(self, save_dir: str | Path, images: None | List[Image] = None) -> None:
        """Writes the point cloud data file in the COLMAP format.

        Reference: https://colmap.github.io/format.html#points3d-txt

        Args:
            self: Scene data to write.
            save_dir: Folder to put the points3D.txt file in.
            images: Optional list of all images for this scene, in order of image index, for color extraction.
        """
        dir_path = Path(save_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        num_pts = self.number_tracks()
        avg_track_length, _ = self.get_track_length_statistics()

        file_path = dir_path / "points3D.txt"
        with open(file_path, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {num_pts}, mean track length: {np.round(avg_track_length, 2)}\n")

            # TODO: assign unique indices to all keypoints (2d points)
            point2d_idx = 0

            for j in range(num_pts):
                track = self.get_track(j)

                r, g, b = (
                    image_utils.get_average_point_color(track, images)
                    if images
                    else (int(track.r), int(track.g), int(track.b))
                )
                _, avg_track_error = reprojection.compute_track_reprojection_errors(self._cameras, track)
                x, y, z = track.point3()
                f.write(f"{j} {x} {y} {z} {r} {g} {b} {np.round(avg_track_error, 2)} ")

                for k in range(track.numberMeasurements()):
                    i, uv_measured = track.measurement(k)
                    f.write(f"{i} {point2d_idx} ")
                f.write("\n")

    def export_as_colmap_text(
        self,
        save_dir: str | Path,
        images: Optional[Sequence[Image]] = None,
        image_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
        image_filenames: Optional[Sequence[str]] = None,
    ) -> None:
        """Emulates the COLMAP option to `Export model as text`.

        Three text files will be saved to disk: "points3D.txt", "images.txt", and "cameras.txt".

        Args:
            save_dir: Folder where text files will be saved.
            images: Optional list of all images for this scene, in order of image index.
            image_shapes: Optional list of image shapes (H, W, C) for each image, required if images is None.
            image_filenames: Optional list of image file names for each image, required if images is None.
        """
        if images is not None:
            image_shapes_to_use = [img.shape for img in images]
            image_file_names_to_use: List[str | None] = [img.file_name for img in images]
        else:
            if image_shapes is None or image_filenames is None:
                raise ValueError("If images is None, image_shapes and image_filenames must be provided.")
            image_shapes_to_use = image_shapes
            image_file_names_to_use = list(image_filenames)

        self.write_cameras(save_dir, image_shapes_to_use)
        self.write_images(save_dir, image_file_names_to_use)
        self.write_points(save_dir, images)
