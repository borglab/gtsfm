"""Loader for datasets used in 1DSFM and Colmap papers.

Authors: Yanwei Du
"""

import glob
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from gtsam import Cal3Bundler, Pose3, PinholeCameraCal3Bundler, Rot3, SfmTrack

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase


logger = logger_utils.get_logger()


class OneDSFMLoader(LoaderBase):
    """Loader for datasets used in 1DSFM and Colmap papers.

    The dataset is collected from internet and only partial images have exif data embedded. It can be downloaded from
    1DSFM [project website](https://www.cs.cornell.edu/projects/1dsfm/) and has the following folder structure:

        - images.SEQ_NAME/SEQ_NAME/images/*.jpg

    Reference:
    [1] Wilson, K., Snavely, N. (2014). Robust Global Translations with 1DSfM. ECCV 2014.
    https://doi.org/10.1007/978-3-319-10578-9_5.
    [2] J. L. Schönberger and J. -M. Frahm, "Structure-from-Motion Revisited," 2016 IEEE Conference on Computer Vision
    and Pattern Recognition (CVPR), 2016, pp. 4104-4113, doi: 10.1109/CVPR.2016.445.
    """

    def __init__(
        self,
        folder: str,
        image_extension: str = "jpg",
        max_resolution: int = 640,
        enable_no_exif: bool = False,
        default_focal_length_factor: float = 1.2,
        gt_dirpath: Optional[str] = None,
    ) -> None:
        """Initializes to load from a specified folder on disk.

        Args:
            folder: Path to base folder which contains a `images` subdirectory.
            image_extension: File extension for the image files. Defaults to 'jpg'.
            max_resolution: Integer representing maximum length of image's short side, i.e.
                the smaller of the height/width of the image. e.g. for 1080p (1920 x 1080),
                max_resolution would be 1080. If the image resolution max(height, width) is
                greater than the max_resolution, it will be downsampled to match the max_resolution.
            enable_no_exif: Whether to read images without exif.
            default_focal_length_factor: Focal length is initialized to default_focal_length_factor * largest dimension
                of image if exif data is not available. The value has not been tuned for performance, 1.2 would be a
                good start.
            gt_dirpath: Path to directory containing 1dsfm ground truth (i.e. Bundler output data and 1dsfm output data).
        """
        super().__init__(max_resolution=max_resolution)
        self._default_focal_length_factor = default_focal_length_factor

        # Fetch all the file names in /images folder.
        search_path = os.path.join(folder, "images", f"*.{image_extension}")

        if enable_no_exif:
            self._image_paths = glob.glob(search_path)
        else:
            (self._image_paths, num_all_imgs) = self.get_images_with_exif(search_path)
            logger.info("Read %d images with exif out of %d in total.", len(self._image_paths), num_all_imgs)

        self._num_imgs = len(self._image_paths)
        if self._num_imgs == 0:
            raise RuntimeError(f"Loader could not find any images with the specified file extension in {search_path}")

        if gt_dirpath is not None:
            self._bundler_data = read_bundler_file(
                bundler_output_fpath=Path(gt_dirpath, "gt_bundle.out"),
                image_list_fpath=Path(gt_dirpath, "list.txt"),
                scene_root=folder,
            )
        else:
            self._bundler_data = None

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return [Path(fpath).name for fpath in self._image_paths]

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            The number of images.
        """
        return self._num_imgs

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution.

        Args:
            index: The index to fetch.

        Returns:
            The image at the query index.

        Raises:
            IndexError: If an out-of-bounds image index is requested.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")
        return io_utils.load_image(self._image_paths[index])

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Bundler]:
        """Get the camera intrinsics at the given index, valid for a full-resolution image.

        Args:
            index: The index to fetch.

        Returns:
            Intrinsics for the given camera.
        """
        # Get intrinsics.
        return io_utils.load_image(self._image_paths[index]).get_intrinsics(
            default_focal_length_factor=self._default_focal_length_factor
        )

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the camera pose (in world coordinates) at the given index.

        The 1DSfM datasets **do** provide ground truth from Bundler.
        See https://github.com/wilsonkl/SfM_Init#file-formats

        Args:
            index: The index to fetch.

        Returns:
            The camera pose w_T_index.
        """
        return self.bundler_data.get_camera(index).pose()


def read_1dsfm_image_list(fpath: str) -> List[str]:
    """Reads in image filenames from a `list.txt` file containing image metadata per row.

    A `list.txt` file should contain a list of all of the images in a dataset, as well as image focal lengths in
    pixels. The format per line is `<image name> 0 <focal length>`, although when the focal length is unknown the
    latter two fields are omitted. SfM Init ignores photos with unknown focal length. The line number of an image
    in this file is its identifying index in the rest of the 1dsfm toolkit. (Note that this list typically includes
    many more images than are in the connected component file also supplied by 1dsfm.)

    See https://github.com/wilsonkl/SfM_Init/blob/fd012ef93462b8623e8d65fa0c6fa95b32270a3c/README.md?plain=1#L128

    Args:
        fpath: Path to file containing image filenames for all images.

    Returns:
        List of image filenames.
    """
    if Path(fpath).name != "list.txt":
        raise ValueError(f"File path to 1dsfm image list should end in `list.txt`, but found {fpath}.")

    image_fnames = []
    with open(image_list_fpath) as f:
        for line in f:
            # Remove carriage return.
            image_fname, *_ = line.rstrip().split(" ")
            image_fnames.append(image_fname)

    return image_fnames


def read_bundler_file(bundler_output_fpath: str, image_list_fpath: str, scene_root: str) -> GtsfmData:
    """Reads a Bundler reconstruction from disk and converts it to a GtsfmData object.

    Reference for file I/O: https://github.com/wilsonkl/SfM_Init/blob/master/sfminit/bundletypes.py#L519

    Args:
        bundler_output_fpath: Path to file containing Bundler output.
        image_list_fpath: Path to file containing image filenames for all images.
        scene_root: Root directory for 1dsfm scene, which should contain `images` subdirectory.

    Returns:
        GtsfmData instance containing Bundler-estimated cameras and tracks.
    """
    if Path(bundler_output_fpath).name != "gt_bundle.out":
        raise ValueError(
            f"File path to Bundler output data should end in `gt_bundle.out`, but found {bundler_output_fpath}."
        )
    image_fnames = read_1dsfm_image_list(fpath=image_list_fpath)

    with open(bundler_output_fpath) as f:
        # Parse the header.
        line = f.readline()
        bundle_version = line.split()[-1]
        if bundle_version != "v0.3":
            raise ValueError(f"Bundle version {bundle_version} not supported.")
        tokens = f.readline().split()
        n_cams = int(tokens[0])
        num_pts = int(tokens[1])
        gtsfm_data = GtsfmData(number_images=n_cams)
        calibrations = []

        # Parse all of the cameras.
        for i in range(n_cams):
            try:
                # Parse intrinsics.
                data = np.fromstring(f.readline(), sep=" ")
                fl = data[0]
                k1 = data[1]
                k2 = data[2]

                # Principal point is not provided, but assume it is located at center of image.
                # image_fpath = Path(scene_root, image_fnames[i])
                # image = io_utils.load_image(img_path=image_fpath)
                # height, width = image.height, image.width
                # u0, v0 = width / 2, height / 2
                u0, v0 = 500, 500

                # Parse camera pose.
                R0 = np.fromstring(f.readline(), sep=" ")
                R1 = np.fromstring(f.readline(), sep=" ")
                R2 = np.fromstring(f.readline(), sep=" ")
                R = np.vstack((R0, R1, R2))
                t = np.fromstring(f.readline(), sep=" ")
                # 1dsfm stores inverse `iTw`
                # See https://github.com/wilsonkl/SfM_Init/blob/master/sfminit/bundletypes.py#L332C51-L332C51
                wTi = Pose3(Rot3(R), t).inverse()

                intrinsics = Cal3Bundler(fl, k1, k2, u0, v0)
                gtsfm_data.add_camera(index=i, camera=PinholeCameraCal3Bundler(wTi, intrinsics))
            except:
                raise ValueError(f"Bundle file format error in camera {i} block")

        # Parse all of the points, each corresponding to 1 track.
        for j in range(num_pts):
            try:
                # Parse 3d coordinate (x,y,z) of the 3d point.
                X = np.fromstring(f.readline(), sep=" ")
                sfm_track = SfmTrack(X)

                color = np.fromstring(f.readline(), sep=" ", dtype=np.uint8)
                data = np.fromstring(f.readline(), sep=" ")
                data = np.reshape(data[1:], (-1, 4))
                observations = []
                # # Loop through track observations.
                # for k in range(data.shape[0]):
                #     image_id = int(data[k, 0])
                #     feature_num = int(data[k, 1])
                #     # Note: `feature_num` is required to get accurate `uv` measurements.
                #     # `uv` should be provided in pixel coordinates, but is typically all 0s.
                #     uv = data[k, 2:4]
                #     sfm_track.addMeasurement(image_id, uv)
                gtsfm_data.add_track(track=sfm_track)
            except:
                raise ValueError(f"Bundle file format error in point {j} block")

        return gtsfm_data


# 1463
scene_root = "/Users/johnlambert/Downloads/Gendarmenmarkt"
image_list_fpath = "/Users/johnlambert/Downloads/Gendarmenmark-gt-2023-08-07/Gendarmenmarkt/list.txt"
bundler_output_fpath = "/Users/johnlambert/Downloads/Gendarmenmark-gt-2023-08-07/Gendarmenmarkt/gt_bundle.out"
gtsfm_data = read_bundler_file(bundler_output_fpath, image_list_fpath, scene_root)

# image_fnames = read_1dsfm_image_list(fpath=image_list_fpath)
# images = [io_utils.load_image(img_path=Path(scene_root, fname)) for fname in image_fnames]
images = []

save_dir = "/Users/johnlambert/Downloads/1dsfm_Gendarmenmarkt_gt_colmap_format_2023_08_07d"
io_utils = io_utils.export_model_as_colmap_text(gtsfm_data=gtsfm_data, images=images, save_dir=save_dir)
