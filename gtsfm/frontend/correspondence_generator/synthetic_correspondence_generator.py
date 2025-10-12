"""Correspondence generator that creates synthetic keypoint correspondences using a 3d mesh.

Authors: John Lambert
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d
import trimesh
from dask.distributed import Client, Future
from gtsam import Pose3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_base import KeypointAggregatorBase
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_dedup import (
    KeypointAggregatorDedup,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.tanks_and_temples_loader import TanksAndTemplesLoader
from gtsfm.products.visibility_graph import VisibilityGraph

logger = logger_utils.get_logger()


class SyntheticCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Pair-wise synthetic keypoint correspondence generator."""

    def __init__(self, dataset_dir: str, scene_name: str, deduplicate: bool = True) -> None:
        """
        Args:
            dataset_dir: Path to where Tanks & Temples dataset is stored.
            scene_name: Name of scene from Tanks & Temples dataset.
            deduplicate: Whether to de-duplicate with a single image the detections received from each image pair.
        """
        self._dataset_root = dataset_dir
        self._scene_name = scene_name
        self._aggregator: KeypointAggregatorBase = (
            KeypointAggregatorDedup() if deduplicate else KeypointAggregatorUnique()
        )

    def generate_correspondences(
        self,
        client: Client,
        images: List[Future],
        visibility_graph: VisibilityGraph,
        num_sampled_3d_points: int = 5000,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences (in parallel).

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images, as futures.
            visibility_graph: The visibility graph defining which image pairs to process.
            num_sampled_3d_points: Number of 3d points to sample from the mesh surface and to project.

        Returns:
            List of keypoints, with one entry for each input image.
            Putative correspondences as indices of keypoints (N,2), for pairs of images (i1,i2).
        """
        dataset_dir = self._dataset_root
        scene_name = self._scene_name

        img_dir = f"{dataset_dir}/{scene_name}"
        poses_fpath = f"{dataset_dir}/{scene_name}_COLMAP_SfM.log"
        lidar_ply_fpath = f"{dataset_dir}/{scene_name}.ply"
        colmap_ply_fpath = f"{dataset_dir}/{scene_name}_COLMAP.ply"
        ply_alignment_fpath = f"{dataset_dir}/{scene_name}_trans.txt"
        bounding_polyhedron_json_fpath = f"{dataset_dir}/{scene_name}.json"
        loader = TanksAndTemplesLoader(
            img_dir=img_dir,
            poses_fpath=poses_fpath,
            lidar_ply_fpath=lidar_ply_fpath,
            ply_alignment_fpath=ply_alignment_fpath,
            bounding_polyhedron_json_fpath=bounding_polyhedron_json_fpath,
            colmap_ply_fpath=colmap_ply_fpath,
        )

        mesh = loader.reconstruct_mesh()

        # Sample random 3d points. This sampling must occur only once, to avoid clusters from repeated sampling.
        pcd = mesh.sample_points_uniformly(number_of_points=num_sampled_3d_points)
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_sampled_3d_points, pcl=pcd)
        sampled_points = np.asarray(pcd.points)

        # TODO(johnwlambert): File Open3d bug to add pickle support for TriangleMesh.
        open3d_mesh_path = tempfile.NamedTemporaryFile(suffix=".obj").name
        open3d.io.write_triangle_mesh(filename=open3d_mesh_path, mesh=mesh)

        loader_future = client.scatter(loader, broadcast=False)

        # TODO(johnwlambert): Remove assumption that image pair shares the same image shape.
        image_height_px, image_width_px, _ = loader.get_image(0).shape

        def apply_synthetic_corr_generator(loader_: LoaderBase, **kwargs) -> Tuple[Keypoints, Keypoints]:
            return generate_synthetic_correspondences_for_image_pair(
                image_height_px=image_height_px, image_width_px=image_width_px, **kwargs
            )

        pairwise_correspondence_futures = {
            (i1, i2): client.submit(
                apply_synthetic_corr_generator,
                loader_future,
                camera_i1=loader.get_camera(index=i1),
                camera_i2=loader.get_camera(index=i2),
                open3d_mesh_fpath=open3d_mesh_path,
                points=sampled_points,
            )
            for i1, i2 in visibility_graph
        }

        pairwise_correspondences: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]] = client.gather(
            pairwise_correspondence_futures
        )

        keypoints_list, putative_corr_idxs_dict = self._aggregator.aggregate(keypoints_dict=pairwise_correspondences)
        return keypoints_list, putative_corr_idxs_dict


def generate_synthetic_correspondences_for_image_pair(
    camera_i1: gtsfm_types.CAMERA_TYPE,
    camera_i2: gtsfm_types.CAMERA_TYPE,
    open3d_mesh_fpath: str,
    points: np.ndarray,
    image_height_px: int,
    image_width_px: int,
) -> Tuple[Keypoints, Keypoints]:
    """Generates synthetic correspondences for image pair.

    Args:
        camera_i1: First camera.
        camera_i2: Second camera.
        open3d_mesh_fpath: Path to saved Open3d mesh.
        points: 3d points sampled from mesh surface.
        image_height_px: Image height, in pixels.
        image_width_px: Image width, in pixels.

    Returns:
        Tuple of `Keypoints` objects, one for each image in the input image pair.
    """
    mesh = open3d.io.read_triangle_mesh(filename=open3d_mesh_fpath)
    trimesh_mesh = load_from_trimesh(open3d_mesh_fpath)

    wTi_list = [camera_i1.pose(), camera_i2.pose()]
    calibrations = [camera_i1.calibration(), camera_i2.calibration()]

    keypoints_i1 = []
    keypoints_i2 = []

    # TODO(johnwlambert): Vectorize this code. On CPU, rays cannot be simultaneously cast against mesh
    # due to RAM limitations.
    for point in points:
        # Try projecting point into each camera. If inside FOV of both and un-occluded by mesh, keep.
        uv_i1 = verify_camera_fov_and_occlusion(camera_i1, point, trimesh_mesh, image_height_px, image_width_px)
        uv_i2 = verify_camera_fov_and_occlusion(camera_i2, point, trimesh_mesh, image_height_px, image_width_px)
        if uv_i1 is not None and uv_i2 is not None:
            keypoints_i1.append(uv_i1)
            keypoints_i2.append(uv_i2)

        visualize = False
        if visualize:
            visualize_ray_to_sampled_mesh_point(camera_i1, point, wTi_list, calibrations, mesh)

    keypoints_i1 = Keypoints(coordinates=np.array(keypoints_i1))
    keypoints_i2 = Keypoints(coordinates=np.array(keypoints_i2))
    print(f"Generated {len(keypoints_i1)} keypoints from sampled {points.shape[0]} 3d points.")
    return keypoints_i1, keypoints_i2


def visualize_ray_to_sampled_mesh_point(
    camera: gtsfm_types.CAMERA_TYPE,
    point: np.ndarray,
    wTi_list: List[Pose3],
    calibrations: List[gtsfm_types.CALIBRATION_TYPE],
    mesh: open3d.geometry.TriangleMesh,
) -> None:
    """Visualizes ray from camera center to 3d point, along with camera frustum, mesh, and 3d point as ball.

    Args:
        camera: Camera to use.
        point: 3d point as (3,) array.
        wTi_list: All camera poses.
        calibrations: Calibration for each camera.
        mesh: 3d surface mesh.
    """
    frustums = open3d_vis_utils.create_all_frustums_open3d(wTi_list, calibrations, frustum_ray_len=0.3)

    # Create line segment to represent ray.
    cam_center = camera.pose().translation()
    ray_dirs = point - camera.pose().translation()
    line_set = _make_line_plot(cam_center, cam_center + ray_dirs)
    # line_set = _make_line_plot(cam_center, camera.backproject(uv_reprojected, depth=1.0))

    # Plot 3d point as red sphere.
    point_cloud = np.reshape(point, (1, 3))
    rgb = np.array([255, 0, 0]).reshape(1, 3).astype(np.uint8)
    spheres = open3d_vis_utils.create_colored_spheres_open3d(point_cloud, rgb, sphere_radius=0.5)

    # Plot all camera frustums and mesh, with sphere and ray line segment.
    open3d.visualization.draw_geometries([mesh] + frustums + spheres + [line_set])


def verify_camera_fov_and_occlusion(
    camera: gtsfm_types.CAMERA_TYPE,
    point: np.ndarray,
    trimesh_mesh: trimesh.Trimesh,
    image_height_px: int,
    image_width_px: int,
) -> Optional[np.ndarray]:
    """Verifies point lies within camera FOV and is unoccluded by mesh faces.

    Args:
        camera: Camera to use.
        point: 3d point as (3,) array.
        trimesh_mesh: Trimesh mesh object to raycast against.
        image_height_px: Image height, in pixels.
        image_width_px: Image width, in pixels.

    Returns:
        2d keypoint as (2,) array.
    """
    # Project to camera.
    uv_reprojected, success_flag = camera.projectSafe(point)
    # Check for projection error in camera.
    if not success_flag:
        return None

    if (
        (uv_reprojected[0] < 0)
        or (uv_reprojected[0] > image_width_px)
        or (uv_reprojected[1] < 0)
        or (uv_reprojected[1] > image_height_px)
    ):
        # Outside of synthetic camera's FOV.
        return None

    # Cast ray through keypoint back towards scene.
    cam_center = camera.pose().translation()
    # Choose an arbitrary depth for the ray direction.
    ray_dirs = point - camera.pose().translation()

    # Returns the location of where a ray hits a surface mesh.
    # keypoint_ind: (M,) array of keypoint indices whose corresponding ray intersected the ground truth mesh.
    # intersections_locations: (M, 3), array of ray intersection locations.
    intersections, keypoint_ind, _ = trimesh_mesh.ray.intersects_location(
        ray_origins=cam_center.reshape(1, 3), ray_directions=ray_dirs.reshape(1, 3), multiple_hits=False
    )

    if intersections.shape[0] > 1 or intersections.shape[0] == 0:
        print(f"Unknown failure: intersections= {intersections} with shape {intersections.shape}")
        return None

    # TODO(johnwlambert): Tune this tolerance threshold to allow looser matches.
    eps = 0.1
    if not np.linalg.norm(intersections[0] - point) < eps:
        # print("Skip occluded: ", intersections[0], ", vs. : ", point)
        # There was a closer intersection along the ray than `point`, meaning `point` is occluded by the mesh.
        return None

    return uv_reprojected


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


def _make_line_plot(point1: np.ndarray, point2: np.ndarray) -> open3d.geometry.LineSet:
    """Plot a line segment from `point1` to `point2` using Open3D."""
    verticals_world_frame = np.array([point1, point2])
    lines = [[0, 1]]
    # Color is in range [0,1]
    color = (0, 0, 1)
    colors = [color for i in range(len(lines))]

    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(verticals_world_frame),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set
