"""Simple loader class that reads any of Carl Olsson's datasets from a folder on disk.

Authors: John Lambert
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.io  # type: ignore
from gtsam import Cal3Bundler, Pose3, SfmTrack  # type: ignore

import gtsfm.utils.io as io_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.image import Image
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.loader.loader_base import LoaderBase


class OlssonLoader(LoaderBase):
    """Simple loader class that reads any of Carl Olsson's datasets from a folder on disk.

    Ref: http://www.maths.lth.se/matematiklth/personal/calle/dataset/dataset.html
    The mat-file contains Olsson's reconstruction. The variable P{i} contains camera i
    and the imnames(i).name contains the name of the corresponding image.
    "U"" are the reconstructed 3D points and "u_uncalib" contains the feature points for each image.
    "u_uncalib" contains two cells; u_uncalib.points{i} contains image points and u_uncalib.index{i}
    contains the indices of the 3D points corresponding to u_uncalib.points{i}.

    Folder layout structure:
    - RGB Images: images/
    - Intrinsics + Extrinsics data (optional): data.mat

    If explicit intrinsics are not provided, the exif data will be used.
    """

    def __init__(
        self,
        folder: str,
        use_gt_intrinsics: bool = True,
        use_gt_extrinsics: bool = True,
        max_frame_lookahead: int = 20,
        max_resolution: int = 760,
    ) -> None:
        """Initializes to load from a specified folder on disk.

        Args:
            folder: The base folder for a given scene.
            use_gt_intrinsics: Whether to use ground truth intrinsics.
            use_gt_extrinsics: Whether to use ground truth extrinsics.
            max_resolution: Integer representing maximum length of image's short side, i.e.
               the smaller of the height/width of the image. e.g. for 1080p (1920 x 1080),
               max_resolution would be 1080. If the image resolution max(height, width) is
               greater than the max_resolution, it will be downsampled to match the max_resolution.
        """
        super().__init__(max_resolution)
        self._folder = folder
        self._use_gt_intrinsics = use_gt_intrinsics
        self._use_gt_extrinsics = use_gt_extrinsics
        self._max_frame_lookahead = max_frame_lookahead

        self._image_paths = io_utils.get_sorted_image_names_in_dir(os.path.join(folder, "images"))
        self._num_imgs = len(self._image_paths)

        if self._num_imgs == 0:
            raise RuntimeError(f"Loader could not find any images with the specified file extension in {folder}")

        cam_matrices_fpath = os.path.join(folder, "data.mat")
        if not Path(cam_matrices_fpath).exists():
            # Not available, so no choice
            self._use_gt_intrinsics = False
            self._use_gt_extrinsics = False
            return

        # Stores camera poses (extrinsics) and intrinsics as 3x4 projection matrices
        # 'P' array will have shape (1,num_imgs), and each element will be a (3,4) matrix
        data = scipy.io.loadmat(cam_matrices_fpath)

        if len(data["P"][0]) != self._num_imgs:
            raise RuntimeError("Number of images found on disk not equal to number of ground truth images.")

        # Each projection matrix is "M"
        # M = K [R | t]
        # in GTSAM notation, M = K @ cTw
        projection_matrices = [data["P"][0][i] for i in range(self._num_imgs)]

        self._K, _ = verification_utils.decompose_camera_projection_matrix(projection_matrices[0])

        self._wTi_list = []
        # First pose is not necessarily identity (in Door it is, but not in Palace of Fine Arts).
        for M in projection_matrices:
            K, wTc = verification_utils.decompose_camera_projection_matrix(M)
            self._wTi_list.append(wTc)

        # GT 3d structure (point cloud)
        self._point_cloud = data["U"].T[:, :3]

    def get_gt_tracks_2d(self) -> List[SfmTrack2d]:
        """Retrieves 2d ground-truth point tracks."""
        tracks_2d = []
        for track_3d in self.get_gt_tracks_3d():
            measurements_2d = []
            for k in range(track_3d.numberMeasurements()):
                i, uv = track_3d.measurement(k)
                measurements_2d.append(SfmMeasurement(i, uv))
            tracks_2d.append(SfmTrack2d(measurements_2d))
        return tracks_2d

    def get_gt_tracks_3d(self) -> List[SfmTrack]:
        """Retrieves 3d ground-truth point tracks."""
        cam_matrices_fpath = os.path.join(self._folder, "data.mat")
        if not Path(cam_matrices_fpath).exists():
            raise ValueError("Ground truth data file missing.")

        data = scipy.io.loadmat(cam_matrices_fpath)

        tracks = []
        # `u_uncalib` contains two cells.
        for j, point in enumerate(self._point_cloud):

            track_3d = SfmTrack(point)
            for i in range(len(self)):

                # u_uncalib.points{i} contains homogeneous image keypoints w/ shape (3, num_visible_points).
                # Transpose to (num_visible_points, 3)
                keypoint_coords = data["u_uncalib"][0, 0][1][i][0].T

                # u_uncalib.index{i} contains the indices of the 3D points corresponding to u_uncalib.points{i}.
                # shape: (1, num_visible_points)
                if j not in data["u_uncalib"][0, 0][2][i][0]:
                    continue

                k = np.argwhere(data["u_uncalib"][0, 0][2][i][0][0] == j)[0]
                uv = np.squeeze(keypoint_coords[k, :2])
                track_3d.addMeasurement(i, uv)

            if track_3d.numberMeasurements() == 0:
                # Empty track in ground truth.
                continue
            tracks.append(track_3d)

        return tracks

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return [Path(fpath).name for fpath in self._image_paths]

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            the number of images.
        """
        return self._num_imgs

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: the index to fetch.

        Raises:
            IndexError: if an out-of-bounds image index is requested.

        Returns:
            Image: the image at the query index.
        """

        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")

        return io_utils.load_image(self._image_paths[index])

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Bundler]:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        if not self._use_gt_intrinsics:
            # get intrinsics from exif
            intrinsics = io_utils.load_image(self._image_paths[index]).get_intrinsics_from_exif()

        else:
            intrinsics = Cal3Bundler(
                fx=min(self._K[0, 0], self._K[1, 1]),
                k1=0,
                k2=0,
                u0=self._K[0, 2],
                v0=self._K[1, 2],
            )
        return intrinsics

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            the camera pose w_T_index.
        """
        if not self._use_gt_extrinsics:
            return None

        wTi = self._wTi_list[index]
        return wTi

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair. idx1 < idx2 is required.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        return super().is_valid_pair(idx1, idx2) and abs(idx1 - idx2) <= self._max_frame_lookahead
