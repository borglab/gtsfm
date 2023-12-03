"""Script to render Tanks & Temples dataset ground truth camera poses and structure, and render the scene to the GUI.

Results must be stored in the Tanks & Temples file format.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d
from gtsam import Cal3Bundler
from dask.distributed import Client, LocalCluster

import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils
from gtsfm.frontend.correspondence_generator.synthetic_correspondence_generator import SyntheticCorrespondenceGenerator
from gtsfm.loader.tanks_and_temples_loader import TanksAndTemplesLoader

import gtsfm.utils.viz as correspondence_viz_utils


TEST_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def view_scene(args: argparse.Namespace) -> None:
    """Read Tanks & Temples Dataset ground truth from JSON, txt, PLY, and .log files and render the scene to the GUI.

    Args:
        args: Rendering options.
    """
    data_root = Path(args.data_root)
    img_dir = data_root / args.scene_name
    poses_fpath = data_root / f"{args.scene_name}_COLMAP_SfM.log"
    ply_alignment_fpath = data_root / f"{args.scene_name}_trans.txt"
    bounding_polyhedron_json_fpath = data_root / f"{args.scene_name}.json"

    lidar_ply_fpath = data_root / f"{args.scene_name}.ply"
    colmap_ply_fpath = data_root / f"{args.scene_name}_COLMAP.ply"

    loader = TanksAndTemplesLoader(
        img_dir=str(img_dir),
        poses_fpath=str(poses_fpath),
        bounding_polyhedron_json_fpath=str(bounding_polyhedron_json_fpath),
        ply_alignment_fpath=str(ply_alignment_fpath),
        lidar_ply_fpath=str(lidar_ply_fpath),
        colmap_ply_fpath=str(colmap_ply_fpath),
    )

    # Both are loaded in the COLMAP world coordinate frame.
    lidar_pcd = loader.get_lidar_point_cloud(downsample_factor=1)
    overlay_point_clouds = False
    if overlay_point_clouds:
        red = np.zeros_like(lidar_pcd.colors)
        red[:, 0] = 1.0
        lidar_pcd.colors = open3d.utility.Vector3dVector(red)

    colmap_pcd = loader.get_colmap_point_cloud(downsample_factor=1)

    calibrations = [loader.get_camera_intrinsics_full_res(0)] * len(loader)
    wTi_list = loader.get_gt_poses()
    frustums = open3d_vis_utils.create_all_frustums_open3d(wTi_list, calibrations, args.frustum_ray_len)

    geometries = frustums + [colmap_pcd] + [lidar_pcd]
    open3d.visualization.draw_geometries(geometries)

    visualize_mesh = True
    if visualize_mesh:
        mesh = loader.reconstruct_mesh()
        open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    visualize_mesh_sampled_points = True
    if visualize_mesh_sampled_points:
        num_sampled_3d_points = 20000
        pcd = mesh.sample_points_uniformly(number_of_points=num_sampled_3d_points)
        # pcd = mesh.sample_points_poisson_disk(number_of_points=num_sampled_3d_points, pcl=pcd)
        open3d.visualization.draw_geometries([pcd])

    visualize_projected_points = False
    if visualize_projected_points:
        i1 = 0
        camera_i1 = loader.get_camera(index=i1)
        img_i1 = loader.get_image_full_res(index=i1)

        # Project LiDAR point cloud into image 1.
        pcd = loader.get_lidar_point_cloud(downsample_factor=10)
        lidar_points = np.asarray(pcd.points)
        keypoints_i1 = _project_points_onto_image(lidar_points, camera_i1)

        # Plot projected LiDAR points.
        plt.imshow(img_i1.value_array.astype(np.uint8))
        plt.scatter(keypoints_i1[:, 0], keypoints_i1[:, 1], 10, color="r", marker=".", alpha=0.007)
        plt.show()

        # Project mesh vertices into image 1.
        mesh = loader.reconstruct_mesh()
        mesh_points = np.asarray(mesh.vertices)
        keypoints_i1_ = _project_points_onto_image(mesh_points, camera_i1)
        # Plot projected mesh points.
        plt.imshow(img_i1.value_array.astype(np.uint8))
        plt.scatter(keypoints_i1_[:, 0], keypoints_i1_[:, 1], 10, color="g", marker=".", alpha=0.007)
        plt.show()

    visualize_synthetic_correspondences = True
    if visualize_synthetic_correspondences:
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)
        client = Client(cluster)

        i1 = 0
        i2 = 4

        synthetic_generator = SyntheticCorrespondenceGenerator(args.data_root, args.scene_name)
        images = [loader.get_image_full_res(index=i) for i in range(i2+1)]
        keypoints_list, corr_idx_dict = synthetic_generator.generate_correspondences(
            client=client,
            images=images,
            image_pairs=[[i1,i2]],
            num_sampled_3d_points=500,
        )

        plot_img = correspondence_viz_utils.plot_twoview_correspondences(
            images[i1],
            images[i2],
            keypoints_list[i1],
            keypoints_list[i2],
            corr_idx_dict[(i1,i2)],
            inlier_mask=np.ones(len(corr_idx_dict[(i1,i2)]), dtype=bool)
        )
        plt.imshow(plot_img.value_array)
        plt.axis('off')
        plt.show()


def _project_points_onto_image(points: np.ndarray, camera: Cal3Bundler) -> np.ndarray:
    """Returns (N,2) array of keypoint coordinates."""
    keypoints = []
    for point in points:
        keypoints.append(camera.projectSafe(point)[0])

    return np.array(keypoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Tanks & Temples dataset w/ Open3d.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join(TEST_DATA_ROOT, "tanks_and_temples_barn"),
        help="Path to data for a specific Tanks & Temples scene.",
    )
    parser.add_argument("--scene_name", type=str, default="Barn")
    parser.add_argument(
        "--frustum_ray_len",
        type=float,
        default=0.1,
        help="Length to extend frustum rays away from optical center "
        + "(increase length for large-scale scenes to make frustums visible)",
    )
    args = parser.parse_args()
    view_scene(args)
