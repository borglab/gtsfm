"""Simple loader class that reads a dataset with metadata formatted in the COLMAP style.

Authors: Travis Driver
"""

import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from gtsam import Cal3Bundler, Pose3, Rot3, Point3, SfmTrack

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase

import thirdparty.colmap.scripts.python.read_write_model as colmap_io
from thirdparty.colmap.scripts.python.read_write_model import Camera as ColmapCamera
from thirdparty.colmap.scripts.python.read_write_model import Image as ColmapImage
from thirdparty.colmap.scripts.python.read_write_model import Point3D as ColmapPoint3D


logger = logger_utils.get_logger()


class AstronetLoader(LoaderBase):
    """Loader class that reads an AstroNet data segment.

    Refs:
    - https://github.com/travisdriver/astronet
    """

    def __init__(
        self,
        data_dir: str,
        use_gt_extrinsics: bool = True,
        use_gt_sfmtracks: bool = False,
        max_frame_lookahead: int = 2,
        max_resolution: int = 1024,
    ) -> None:
        """Initialize loader from a specified segment directory (data_dir) on disk.

        <data_dir>/
             ├── images/: undistorted grayscale images
             ├── cameras.bin: camera calibrations (see https://colmap.github.io/format.html#cameras-txt)
             ├── images.bin: 3D poses and 2D tracks (see https://colmap.github.io/format.html#images-txt)
             └── points3D.bin: 3D tracks (see https://colmap.github.io/format.html#points3d-txt)


        Args:
            data_dir: path to directory containing the COLMAP-formatted data: cameras.bin, images.bin, and points3D.bin
            use_gt_extrinsics (optional): whether to use ground truth extrinsics. Used only for comparison with
                reconstructed values.
            use_gt_sfmtracks (optional): whether to use ground truth tracks. Used only for comparison with reconstructed
                values.
            max_frame_lookahead (optional): maximum number of consecutive frames to consider for
                matching/co-visibility. Any value of max_frame_lookahead less than the size of
                the dataset assumes data is sequentially captured.
            max_resolution: integer representing maximum length of image's short side, i.e.
               the smaller of the height/width of the image. e.g. for 1080p (1920 x 1080),
               max_resolution would be 1080. If the image resolution max(height, width) is
               greater than the max_resolution, it will be downsampled to match the max_resolution.

        Raises:
            FileNotFoundError if `data_dir` doesn't exist or image path does not exist.
            RuntimeError if ground truth camera calibrations not provided.
        """
        super().__init__(max_resolution)
        self._use_gt_extrinsics = use_gt_extrinsics
        self._use_gt_sfmtracks = use_gt_sfmtracks
        self._max_frame_lookahead = max_frame_lookahead

        # Use COLMAP model reader to load data and convert to GTSfM format.
        if not Path(data_dir).exists():
            raise FileNotFoundError("No data found at %s." % data_dir)
        cameras, images, points3d = colmap_io.read_model(path=data_dir, ext=".bin")
        self._calibrations, self._wTi_list, img_fnames, self._sfmtracks = self.colmap2gtsfm(
            cameras, images, points3d, load_sfmtracks=use_gt_sfmtracks
        )

        # Camera intrinsics are currently required due to absence of EXIF data and diffculty in approximating focal
        # length (usually 10000 to 100000 pixels).
        if self._calibrations is None:
            raise RuntimeError("Camera intrinsics cannot be None.")

        if self._wTi_list is None and self._use_gt_extrinsics:
            raise RuntimeError("Ground truth extrinsic data requested but missing.")

        if self._sfmtracks is None and self._use_gt_sfmtracks:
            raise RuntimeError("Ground truth SfMTrack data requested but missing.")
        self.num_sfmtracks = len(self._sfmtracks) if self._sfmtracks is not None else 0

        # Prepare image paths
        self._image_paths = []
        for img_fname in img_fnames:
            img_fpath = os.path.join(data_dir, "images", img_fname)
            if not Path(img_fpath).exists():
                raise FileNotFoundError(f"Could not locate image at {img_fpath}.")
            self._image_paths.append(img_fpath)

        self._num_imgs = len(self._image_paths)
        logger.info("AstroNet loader found and loaded %d images and %d tracks.", self._num_imgs, self.num_sfmtracks)

    @staticmethod
    def colmap2gtsfm(
        cameras: Dict[int, ColmapCamera],
        images: Dict[int, ColmapImage],
        points3D: Dict[int, ColmapPoint3D],
        load_sfmtracks: bool = False,
    ) -> Tuple[List[Cal3Bundler], List[Pose3], List[str], Optional[List[Point3]]]:
        """Converts COLMAP-formatted variables to GTSfM format.

        Args:
            cameras: dictionary of COLMAP-formatted Cameras
            images: dictionary of COLMAP-formatted Images
            points3D: dictionary of COLMAP-formatted Point3Ds
            return_tracks (optional): whether or not to return tracks

        Returns:
            cameras_gtsfm: list of N camera calibrations corresponding to the N images in images_gtsfm
            images_gtsfm: list of N camera poses when each image was taken
            img_fnames: file names of images in images_gtsfm
            sfmtracks_gtsfm: tracks of points in points3D
        """
        # Note: Assumes input cameras use `PINHOLE` model
        if len(images) == 0 and len(cameras) == 0:
            raise RuntimeError("No Image or Camera data provided to loader.")
        cameras_gtsfm, images_gtsfm, img_fnames = [], [], []
        image_id_to_idx = {}  # keeps track of discrepencies between `image_id` and List index.
        for idx, img in enumerate(images.values()):
            images_gtsfm.append(Pose3(Rot3(img.qvec2rotmat()), img.tvec).inverse())
            img_fnames.append(img.name)
            fx, _, cx, cy = cameras[img.camera_id].params[:4]
            cameras_gtsfm.append(Cal3Bundler(fx, 0.0, 0.0, cx, cy))
            image_id_to_idx[img.id] = idx

        if len(points3D) == 0 and load_sfmtracks:
            raise RuntimeError("No SfMTrack data provided to loader.")
        sfmtracks_gtsfm = None
        if len(points3D) > 0 and load_sfmtracks:
            sfmtracks_gtsfm = []
            for point3D in points3D.values():
                sfmtrack = SfmTrack(point3D.xyz)
                for (image_id, point2d_idx) in zip(point3D.image_ids, point3D.point2D_idxs):
                    sfmtrack.add_measurement(image_id_to_idx[image_id], images[image_id].xys[point2d_idx])
                sfmtracks_gtsfm.append(sfmtrack)

        return cameras_gtsfm, images_gtsfm, img_fnames, sfmtracks_gtsfm

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
            raise IndexError(f"Image index {index} is invalid")

        img = io_utils.load_image(self._image_paths[index])
        return img

    def get_camera_intrinsics_full_res(self, index: int) -> Cal3Bundler:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            the index to fetch.

        Returns:
            intrinsics for the given camera.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")
        intrinsics = self._calibrations[index]
        logger.info("Loading ground truth calibration.")

        return intrinsics

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            pose for the given camera.
        """
        if not self._use_gt_extrinsics or self._wTi_list is None:
            return None

        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")

        wTi = self._wTi_list[index]
        return wTi

    def get_sfmtrack(self, index: int) -> Optional[SfmTrack]:
        """Get the SfmTracks(s) (in world coordinates) at the given index.

        Args:
            index: the index to fetch.

        Returns:
            SfmTrack at index.
        """
        if not self._use_gt_sfmtracks or self._sfmtracks is None:
            return None

        if index < 0 or index >= len(self._sfmtracks):
            raise IndexError(f"Track3D index {index} is invalid")

        sfmtrack = self._sfmtracks[index]
        return sfmtrack

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        return super().is_valid_pair(idx1, idx2) and abs(idx1 - idx2) <= self._max_frame_lookahead
