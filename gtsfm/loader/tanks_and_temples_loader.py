"""Loader for the Tanks & Temples dataset.

See https://www.tanksandtemples.org/download/ for more information.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import open3d
import trimesh
from gtsam import Cal3Bundler, Rot3, Pose3

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase
import gtsfm.utils.geometry_comparisons as geom_comp_utils
import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils

logger = logger_utils.get_logger()


_DEFAULT_IMAGE_HEIGHT_PX = 1080
_DEFAULT_IMAGE_WIDTH_PX = 1920


class TanksAndTemplesLoader(LoaderBase):
    def __init__(
        self,
        img_dir: str,
        poses_fpath: str,
        bounding_polyhedron_json_fpath: str,
        ply_alignment_fpath: str,
        lidar_ply_fpath: Optional[str] = None,
        colmap_ply_fpath: Optional[str] = None,
        max_resolution: int = 1080,
        max_num_images: Optional[int] = None,
    ) -> None:
        """Initializes image file paths and GT camera poses.

        There are two coordinate frames -- the COLMAP coordinate frame, and the GT LiDAR coordinate frame.
        We move everything to the COLMAP coordinate frame.

        Args:
            img_dir: Path to where images of a single scene are stored.
            poses_fpath: Path to .log file containing COLMAP-reconstructed camera poses.
            bounding_polyhedron_json_fpath: Path to JSON file containing specification of bounding polyhedron
                to crop the COLMAP reconstructed point cloud.
            ply_alignment_fpath: The alignment text file contains the transformation matrix to align the COLMAP
                reconstruction to the corresponding ground-truth point cloud.
            lidar_ply_fpath: Path to LiDAR scan, in PLY format. Omitted for unit tests.
            colmap_ply_fpath: Path to COLMAP reconstructed point cloud, in PLY format. Omitted for unit tests.
            max_num_images: Maximum number of images to use for reconstruction.
        """
        super().__init__(max_resolution)
        self.lidar_ply_fpath = lidar_ply_fpath
        self.colmap_ply_fpath = colmap_ply_fpath
        self.bounding_polyhedron_json_fpath = bounding_polyhedron_json_fpath
        self._image_paths = sorted(list(Path(img_dir).glob("*.jpg")))

        # Load the Sim(3), not SE(3), transform between LiDAR global coordinate frame and COLMAP global coordinate
        # frame.
        self.lidar_Sim3_colmap = np.loadtxt(fname=ply_alignment_fpath)

        self._use_gt_extrinsics = True
        # The reconstructions are made with an "out of the box" COLMAP configuration and are available as *.ply
        # files together with the camera poses (stored in *.log file format).
        colmapTi_gt_dict = _parse_redwood_data_log_file(poses_fpath)
        self.wTi_gt_dict = {k: (colmapTi) for k, colmapTi in colmapTi_gt_dict.items()}

        if max_num_images is not None:
            # Optionally artificially truncate the size of the dataset.
            self._num_imgs = max_num_images
        else:
            self._num_imgs = len(self.wTi_gt_dict)

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            The number of images.
        """
        return self._num_imgs

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return [Path(fpath).name for fpath in self._image_paths]

    def get_image_fpath(self, index: int = 0) -> Path:
        """ """
        return self._image_paths[index]

    def get_image_full_res(self, index: int) -> Image:
        """Gets the image at the given index, at full resolution.

        Args:
            index: The index to fetch.

        Returns:
            The image at the query index.

        Raises:
            IndexError: If an out-of-bounds image index is requested.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")

        img = io_utils.load_image(self._image_paths[index])

        # All should have shape (1080, 1920, 3)
        if img.height != _DEFAULT_IMAGE_HEIGHT_PX:
            raise ValueError(f"Images from the Tanks&Temples dataset should have height {_DEFAULT_IMAGE_HEIGHT_PX} px.")
        if img.width != _DEFAULT_IMAGE_WIDTH_PX:
            raise ValueError(f"Images from the Tanks&Temples dataset should have width {_DEFAULT_IMAGE_WIDTH_PX} px.")
        return img

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Bundler]:
        """Gets the camera intrinsics at the given index, valid for a full-resolution image.

        Note: Tanks & Temples does not release the intrinsics from the COLMAP reconstruction.

        Args:
            The index to fetch.

        Returns:
            Intrinsics for the given camera.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")

        # Use synthetic camera.
        H = _DEFAULT_IMAGE_HEIGHT_PX
        W = _DEFAULT_IMAGE_WIDTH_PX
        # Principal point offset:
        cx = W / 2
        cy = H / 2
        # Focal length:
        fx = 0.7 * W

        # TODO(johnwlambert): Add Sony A7SM2 to sensor DB (35.6 x 23.8 mm), and get intrinsics from exif.
        # intrinsics = io_utils.load_image(self._image_paths[index]).get_intrinsics_from_exif()

        intrinsics = Cal3Bundler(fx, k1=0.0, k2=0.0, u0=cx, v0=cy)
        return intrinsics

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Gets the GT camera pose (in world coordinates) at the given index.

        Args:
            index: The index to fetch.

        Returns:
            The camera pose w_T_index.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Image index is invalid")

        if not self._use_gt_extrinsics:
            return None

        # wTi = self._wTi_list[index]
        wTi = self.wTi_gt_dict[index]
        if not geom_comp_utils.is_valid_SO3(wTi.rotation()):
            raise ValueError("Given GT rotation is not a member of SO(3) and GT metrics will be incorrect.")
        return wTi

    def get_lidar_point_cloud(self, downsample_factor: int = 10) -> open3d.geometry.PointCloud:
        """Returns ground-truth point cloud, captured using an industrial laser scanner.

        Move all LiDAR points to the COLMAP frame.
        """
        if not Path(self.lidar_ply_fpath).exists():
            raise ValueError("")
        pcd = open3d.io.read_point_cloud(self.lidar_ply_fpath)
        points, rgb = open3d_vis_utils.convert_colored_open3d_point_cloud_to_numpy(pointcloud=pcd)
        points = points[::downsample_factor]
        rgb = rgb[::downsample_factor]

        lidar_Sim3_colmap = _create_Sim3_from_tt_dataset_alignment_transform(self.lidar_Sim3_colmap)
        colmap_Sim3_lidar = np.linalg.inv(lidar_Sim3_colmap)
        # Transform LiDAR points to COLMAP coordinate frame.
        points = transform_point_cloud_vectorized(points, colmap_Sim3_lidar)
        pcd = open3d_vis_utils.create_colored_point_cloud_open3d(point_cloud=points, rgb=rgb)
        return pcd

    def get_colmap_point_cloud(self, downsample_factor: int = 1) -> open3d.geometry.PointCloud:
        """Returns COLMAP-reconstructed point cloud."""
        if not Path(self.colmap_ply_fpath).exists():
            raise ValueError("")
        pcd = open3d.io.read_point_cloud(self.colmap_ply_fpath)
        points, rgb = open3d_vis_utils.convert_colored_open3d_point_cloud_to_numpy(pointcloud=pcd)
        points = points[::downsample_factor]
        rgb = rgb[::downsample_factor]
        pcd = open3d_vis_utils.create_colored_point_cloud_open3d(point_cloud=points, rgb=rgb)
        return pcd


def crop_points_to_bounding_polyhedron(pcd: open3d.geometry.PointCloud, json_fpath: str) -> open3d.geometry.PointCloud:
    """Crops a point cloud according to JSON-specified polyhedron crop bounds.

    Args:
        pcd: Input point cloud.
        json_fpath: Path to JSON file containing crop specification, including 'orthogonal_axis',
            'axis_min', 'axis_max', 'bounding_polygon'.

    Returns:
        Cropped point cloud, according to `SelectionPolygonVolume`.
    """
    vol = open3d.visualization.read_selection_polygon_volume(json_fpath)
    cropped_pcd = vol.crop_point_cloud(pcd)
    return cropped_pcd


def _parse_redwood_data_log_file(log_fpath: str) -> Dict[int, Pose3]:
    """Parses camera poses provided in the "Redwood Data" log file format.

    Trajectory File (.log) format reference: http://redwood-data.org/indoor/fileformat.html

    Args:
        log_fpath: Path to .log file containing COLMAP-reconstructed camera poses.

    Returns:
        Mapping from camera index to camera pose.
    """
    wTi_gt_dict = {}
    with open(log_fpath, "r") as f:
        data = f.readlines()

    # Every five lines are an item.
    if len(data) % 5 != 0:
        raise ValueError("Invalid log file format; must contain grouped 5-line entries.")
    num_images = len(data) // 5

    # The first line contains three numbers which store metadata.
    # In a typical camera pose trajectory file, the third number of the metadata is the frame number of the
    # corresponding depth image (starting from 1).
    parse_metadata = lambda x: int(x.split(" ")[0])
    parse_matrix_row = lambda x: [float(val) for val in x.split(" ")]

    for i in range(num_images):
        metadata = data[i * 5]
        cam_idx = parse_metadata(metadata)
        # The other four lines make up the homogeneous transformation matrix.
        lines = data[(i * 5) + 1 : (i + 1) * 5]
        # The transformation matrix maps a point from its local coordinates (in homogeneous form)
        # to the world coordinates: p_w = wTi * p_i
        wTi_gt = np.array([parse_matrix_row(line) for line in lines])
        wTi_gt_dict[cam_idx] = Pose3(wTi_gt)
    return wTi_gt_dict


def _make_line_plot(point1: np.ndarray, point2: np.ndarray) -> open3d.geometry.LineSet:
    """ """
    verts_worldfr = np.array([point1, point2])
    lines = [[0, 1]]
    # color is in range [0,1]
    color = (0, 0, 1)
    colors = [color for i in range(len(lines))]

    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(verts_worldfr),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set


def load_from_trimesh(mesh_path: str) -> trimesh.Trimesh:
    """Read in scene mesh as Trimesh object."""
    if not Path(mesh_path).exists():
        raise FileNotFoundError(f"No mesh found at {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
    logger.info(
        "Tanks & Temples loader read in mesh with %d vertices and %d faces.",
        mesh.vertices.shape[0],
        mesh.faces.shape[0],
    )
    return mesh


def _create_Sim3_from_tt_dataset_alignment_transform(lidar_Sim3_colmap: np.ndarray) -> np.ndarray:
    """Create member of Sim(3) matrix group given Tanks & Temples dataset alignment transform.

    Args:
        lidar_Sim3_colmap: Sim(3) transformation in non-standard form (invalid, non-group form).

    Returns:
        Sim(3) transformation matrix in group form. See Section 6.1 of https://www.ethaneade.com/lie.pdf
    """
    T = lidar_Sim3_colmap
    # Disentangle scale factor from 3d rotation.
    R = Rot3.ClosestTo(lidar_Sim3_colmap[:3, :3]).matrix()
    s = lidar_Sim3_colmap[0, 0] / R[0, 0]
    t = lidar_Sim3_colmap[:3, 3] / s

    # Create 4x4 matrix in group notation.
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1 / s
    return T


def transform_point_cloud_vectorized(points_b: np.ndarray, aTb: np.ndarray) -> np.ndarray:
    """Given points in `b` frame, transform them to `a `frame.

    Args:
        points_b: (N,3) points in `b` frame.
        aTb: 4x4 transformation matrix, a member of either SE(3) or Sim(3).

    Returns:
        (N,3) points in `a` frame.
    """
    N, _ = points_b.shape
    points_b = np.concatenate([points_b, np.ones((N, 1))], axis=1)
    points_a = aTb @ points_b.T
    points_a = points_a.T
    # Remove homogeonous coordinate.
    points_a = points_a[:, :3] / points_a[:, 3][:, np.newaxis]
    return points_a
