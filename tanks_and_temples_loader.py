"""Loader for the Tanks & Temples dataset.

See https://www.tanksandtemples.org/download/ for more information.
"""

import os
from enum import auto, Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d
from gtsam import Cal3Bundler, Rot3, Pose3
from dask.distributed import Client, Future

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.loader.loader_base import LoaderBase


_DEFAULT_IMAGE_HEIGHT_PX = 1080
_DEFAULT_IMAGE_WIDTH_PX = 1920


class MeshReconstructionType(Enum):
    """Mesh reconstruction algorithm choices.

    The alpha shape [Edelsbrunner1983] is a generalization of a convex hull.
    See Edelsbrunner and D. G. Kirkpatrick and R. Seidel: On the shape of a set of points in the plane, 1983.
    https://www.cs.jhu.edu/~misha/Fall13b/Papers/Edelsbrunner93.pdf

    The ball pivoting algorithm (BPA) [Bernardini1999].
    See https://www.cs.jhu.edu/~misha/Fall13b/Papers/Bernardini99.pdf

    The Poisson surface reconstruction method [Kazhdan2006].
    See: M.Kazhdan and M. Bolitho and H. Hoppe: Poisson surface reconstruction, Eurographics, 2006.
    See https://hhoppe.com/poissonrecon.pdf
    """

    ALPHA_SHAPE = auto()
    BALL_PIVOTING = auto()
    POISSION_SURFACE = auto()


class TanksAndTemplesLoader(LoaderBase):
    def __init__(
        self,
        img_dir: str,
        poses_fpath: str,
        ply_fpath: str,
        ply_alignment_fpath: str,
        bounding_polyhedron_json_fpath: str,
    ) -> None:
        """Initializes image file paths and GT camera poses.

        There are two coordinate frames -- the COLMAP coordinate frame, and the GT LiDAR coordinate frame.
        We move everything to the GT LiDAR coordinate frame.

        Args:
            img_dir: Path to where images of a single scene are stored.
            poses_fpath: Path to .log file containing COLMAP-reconstructed camera poses.
            ply_fpath:
            ply_alignment_fpath: The alignment text file contains the transformation matrix to align the COLMAP
                reconstruction to the corresponding ground-truth point cloud.
            bounding_polyhedron_json_fpath: Path to JSON file containing specification of bounding polyhedron
                to crop the COLMAP reconstructed point cloud.
        """
        self.ply_fpath = ply_fpath
        self.bounding_polyhedron_json_fpath = bounding_polyhedron_json_fpath
        self._image_paths = list(Path(img_dir).glob("*.jpg"))

        T = np.loadtxt(fname=ply_alignment_fpath)
        self.T = Pose3(r=Rot3(T[:3,:3]), t=T[:3,3])
        print(self.T)

        self._use_gt_extrinsics = True
        # The reconstructions are made with an "out of the box" COLMAP configuration and are available as *.ply
        # files together with the camera poses (stored in *.log file format).
        colmapTi_gt_dict = _parse_redwood_data_log_file(poses_fpath)
        self.wTi_gt_dict = {k: self.T.compose(colmapTi) for k, colmapTi in colmapTi_gt_dict.items() }

        self._num_imgs = len(self.wTi_gt_dict)

    def __len__(self) -> int:
        """The number of images in the dataset.

        Returns:
            The number of images.
        """
        return self._num_imgs

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

        # All should have shape (1080, 1920, 3)

        img = io_utils.load_image(self._image_paths[index])

        if img.height != _DEFAULT_IMAGE_HEIGHT_PX:
            raise ValueError('')
        if img.width != _DEFAULT_IMAGE_WIDTH_PX:
            raise ValueError('')
        return img

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3Bundler]:
        """Gets the camera intrinsics at the given index, valid for a full-resolution image.

        Sony A7SM2, 35.6 x 23.8 mm

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

        # if not self._use_gt_intrinsics:
        # # get intrinsics from exif
        # TODO(johnwlambert): Add Sony A7SM2 to sensor DB.
        #intrinsics = io_utils.load_image(self._image_paths[index]).get_intrinsics_from_exif()
        #import pdb; pdb.set_trace()
        # else:
        # intrinsics = self._calibrations[index]

        intrinsics = Cal3Bundler(fx, k1=0.0, k2=0.0, u0=cx, v0=cy)
        return intrinsics

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Gets the camera pose (in world coordinates) at the given index.

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
        return wTi

    def get_lidar_point_cloud(self) -> open3d.geometry.PointCloud:
        """Returns ground-truth point cloud, captured using an industrial laser scanner."""
        return open3d.io.read_point_cloud(self.ply_fpath)

    def reconstruct_mesh(
        self,
        point_downsample_factor: int = 10,
        crop_by_polyhedron: bool = True,
        reconstruction_algorithm: MeshReconstructionType = MeshReconstructionType.ALPHA_SHAPE,
    ) -> open3d.geometry.TriangleMesh:
        """Reconstructs mesh from LiDAR PLY file.

        Args:
        point_downsample_factor
        crop_by_polyhedron: Whether to crop by a manually specified polyhedron, vs. simply
        by range from global origin.

        Returns:
        Reconstructed mesh.
        """
        pcd = self.get_lidar_point_cloud()
        pcd = pcd.transform(self.T)
        if crop_by_polyhedron:
            pass
            pcd = crop_points_to_bounding_polyhedron(pcd, self.bounding_polyhedron_json_fpath)

        points, rgb = convert_colored_open3d_point_cloud_to_numpy(pcd)
        if not crop_by_polyhedron:
            max_radius = 4.0
            valid = np.linalg.norm(points, axis=1) < max_radius
            points = points[valid]
            rgb = rgb[valid]
        points = points[::point_downsample_factor]
        rgb = rgb[::point_downsample_factor]
        pcd = create_colored_point_cloud_open3d(points, rgb)
        pcd.estimate_normals()

        if reconstruction_algorithm == MeshReconstructionType.ALPHA_SHAPE:
            alpha = 0.5 # 0.1  # 0.03
            print(f"alpha={alpha:.3f}")
            mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            mesh.compute_vertex_normals()

        elif reconstruction_algorithm == MeshReconstructionType.BALL_PIVOTING:
            radii = [0.005, 0.01, 0.02, 0.04]
            rec_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, open3d.utility.DoubleVector(radii)
            )

        elif reconstruction_algorithm == MeshReconstructionType.POISSION_SURFACE:
            mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        return mesh

    def generate_synthetic_correspondences(
        self,
        images: List[Future],
        image_pairs: List[Tuple[int, int]],
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Generates synthetic correspondences from virtual cameras and a ground-truth mesh.

        TODO: Make this a CorrespondenceGenerator class.

        Returns:
            List of keypoints, one entry for each input image.
            Putative correspondences as indices of keypoints, for pairs of images. Mapping from image pair
                (i1,i2) to delayed task to compute putative correspondence indices. Correspondence indices
                are represented by an array of shape (K,2), for K correspondences.
        """
        mesh = self.reconstruct_mesh()

        camera_dict = {}
        import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils

        keypoints_list = []
        putative_corr_idxs_dict = {}

        for (i1, i2) in image_pairs:
            if i1 not in camera_dict:
                camera_dict[i1] = self.get_camera(index=i1)
            if i2 not in camera_dict:
                camera_dict[i2] = self.get_camera(index=i2)

            # Sample a random 3d point.
            pcd = mesh.sample_points_uniformly(number_of_points=2500)
            #pcd = mesh.sample_points_poisson_disk(number_of_points=500, pcl=pcd)
            points = np.asarray(pcd.points)

            wTi_list = [camera_dict[i1].pose(), camera_dict[i2].pose()]
            calibrations = [camera_dict[i1].calibration(), camera_dict[i1].calibration()]

            # TODO(johnwlambert): Vectorize this code.
            for point in points:

                import pdb; pdb.set_trace()
                point_cloud = np.reshape(point, (1,3))
                rgb = np.array([255,0,0]).reshape(1,3).astype(np.uint8)

                # Visualize two frustums, point, and mesh
                frustums = open3d_vis_utils.create_all_frustums_open3d(wTi_list, calibrations, frustum_ray_len=10)
                spheres = open3d_vis_utils.create_colored_spheres_open3d(
                    point_cloud, rgb, sphere_radius=4.0
                )
                open3d.visualization.draw_geometries([mesh, frustums, points])

                # Try projecting into each camera. If inside FOV of both, keep.
                for i in [i1,i2]:
                    # Get the camera associated with the measurement.
                    camera = camera_dict[i]
                    # Project to camera.
                    uv_reprojected, success_flag = camera.projectSafe(point3d)
                    # Check for projection error in camera.
                    if not success_flag:
                        continue

            # Raytrace backwards.
            src = np.repeat(gt_camera.pose().translation().reshape((-1, 3)), num_kpts, axis=0)  # At_i1A
            drc = np.asarray([gt_camera.backproject(keypoints.coordinates[i], depth=1.0) - src[i, :] for i in range(num_kpts)])
            # Return unique cartesian locations where rays hit the mesh.
            # keypoint_ind: (M,) array of keypoint indices whose corresponding ray intersected the ground truth mesh.
            # intersections_locations: (M, 3), array of ray intersection locations.
            intersections, keypoint_ind, _ = gt_scene_mesh.ray.intersects_location(
                ray_origins=src,
                ray_directions=drc,
                multiple_hits=False
            )
            
            putative_corr_idxs_dict[(i1,i2)] = None

        return keypoints_list, putative_corr_idxs_dict


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


def create_colored_point_cloud_open3d(point_cloud: np.ndarray, rgb: np.ndarray) -> open3d.geometry.PointCloud:
    """Render a point cloud as individual colored points, using Open3d.

    Args:
    point_cloud: array of shape (N,3) representing 3d points.
    rgb: uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255].

    Returns:
    pcd: Open3d geometry object representing a colored 3d point cloud.
    """
    colors = rgb.astype(np.float64) / 255

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    return pcd


def convert_colored_open3d_point_cloud_to_numpy(
pointcloud: open3d.geometry.PointCloud,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
    pointcloud

    Returns:
    points
    rgb
    """
    points = np.asarray(pointcloud.points)
    rgb = np.asarray(pointcloud.colors)
    # open3d stores the colors as [0,1] floats.
    rgb = (rgb * 255).astype(np.uint8)
    return points, rgb


img_dir = '/Users/johnlambert/Downloads/Barn'
log_fpath = '/Users/johnlambert/Downloads/Barn_COLMAP_SfM.log'
ply_fpath = '/Users/johnlambert/Downloads/Barn.ply'
#ply_fpath = '/Users/johnlambert/Downloads/Barn_COLMAP.ply'
ply_alignment_fpath = '/Users/johnlambert/Downloads/Barn_trans.txt'
bounding_polyhedron_json_fpath = '/Users/johnlambert/Downloads/Barn.json'

# img_dir = "/Users/johnlambert/Downloads/Truck"
# log_fpath = "/Users/johnlambert/Downloads/Truck_COLMAP_SfM.log"
# ply_fpath = "/Users/johnlambert/Downloads/Truck.ply"
# ply_alignment_fpath = "/Users/johnlambert/Downloads/Truck_trans.txt"
# bounding_polyhedron_json_fpath = "/Users/johnlambert/Downloads/Truck.json"

loader = TanksAndTemplesLoader(
    img_dir=img_dir,
    poses_fpath=log_fpath,
    ply_fpath=ply_fpath,
    ply_alignment_fpath=ply_alignment_fpath,
    bounding_polyhedron_json_fpath=bounding_polyhedron_json_fpath,
)

intrinsics = loader.get_camera_intrinsics_full_res(index=0)

import gtsfm.utils.io as io_utils


# result = loader.generate_synthetic_correspondences(
#     images = [],
#     image_pairs = [(0,1)]
# ) 



# pcd = io_utils.read_point_cloud_from_ply(ply_fpath)

pcd = loader.get_lidar_point_cloud()
#open3d.visualization.draw_geometries([pcd])

# mesh = loader.reconstruct_mesh()
#open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils

geometries = [pcd] # [mesh]
for index in loader.wTi_gt_dict.keys():
    wTc = loader.get_camera_pose(index)
    #import pdb; pdb.set_trace()
    line_sets = open3d_vis_utils.draw_coordinate_frame(wTc=wTc, axis_length=1.0)
    geometries.extend(line_sets)

    if index % 10 == 0:
        open3d.visualization.draw_geometries(geometries)

"""
Unit Test
(Pdb) p wTi_gt[0]
array([[-0.43321999, -0.05555365, -0.89957447,  3.24710662],
[ 0.05678138,  0.99443357, -0.08875668,  0.14032715],
[ 0.89949781, -0.08953024, -0.42765409,  0.55723886],
[ 0.        ,  0.        ,  0.        ,  1.        ]])
"""


i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, two_view_reports_dict = {}, {}, {}, {}

"""
two_view_results_dict = run_two_view_estimator_as_futures(
    client,
    scene_optimizer.two_view_estimator,
    keypoints_list,
    putative_corr_idxs_dict,
    intrinsics,
    loader.get_relative_pose_priors(image_pair_indices),
    loader.get_gt_cameras(),
    gt_scene_mesh=self.loader.get_gt_scene_trimesh(),
)
"""

# 




