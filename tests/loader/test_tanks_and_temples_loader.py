"""Unit tests for Tanks & Temple dataset loader.

Author: John Lambert
"""

import unittest
from pathlib import Path

import numpy as np
import open3d
from gtsam import Rot3, Unit3

import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils
from gtsfm.loader.tanks_and_temples_loader import TanksAndTemplesLoader
from gtsfm.frontend.verifier.loransac import LoRansac
import gtsfm.utils.geometry_comparisons as geom_comp_utils


_TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "tanks_and_temples_barn"


class TanksAndTemplesLoaderTest(unittest.TestCase):

    def setUp(self) -> None:

        img_dir = _TEST_DATA_ROOT / 'Barn'
        poses_fpath = _TEST_DATA_ROOT / 'Barn_COLMAP_SfM.log'
        lidar_ply_fpath = _TEST_DATA_ROOT / 'Barn.ply'
        colmap_ply_fpath = scene_data_dir / 'Barn_COLMAP.ply'
        ply_alignment_fpath = _TEST_DATA_ROOT / 'Barn_trans.txt'
        bounding_polyhedron_json_fpath = _TEST_DATA_ROOT / 'Barn.json'

        self.loader = TanksAndTemplesLoader(
            img_dir=img_dir,
            poses_fpath=poses_fpath,
            lidar_ply_fpath=lidar_ply_fpath,
            ply_alignment_fpath=ply_alignment_fpath,
            bounding_polyhedron_json_fpath=bounding_polyhedron_json_fpath,
            colmap_ply_fpath=colmap_ply_fpath,
        )

    def test_get_camera_intrinsics_full_res(self) -> None:
        """ """
        intrinsics = self.loader.get_camera_intrinsics_full_res(index=0)

    def test_get_camera_pose(self) -> None:
        """ """
        wTi = self.loader.get_camera_pose(index=0)
        """
        Unit Test
        (Pdb) p wTi_gt[0]
        array([[-0.43321999, -0.05555365, -0.89957447,  3.24710662],
        [ 0.05678138,  0.99443357, -0.08875668,  0.14032715],
        [ 0.89949781, -0.08953024, -0.42765409,  0.55723886],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
        """

    def test_get_image_fpath(self) -> None:
        """ """
        fpath = self.loader.get_image_fpath(index = 0)
        expected_fpath = '/Users/johnlambert/Downloads/Tanks_and_Temples_Barn_410/Barn/000001.jpg'
        assert fpath == expected_fpath


R = Rot3(np.eye(3))
result = geom_comp_utils.is_valid_SO3(R)
assert result

R = Rot3(np.array([
    [3.85615, 0.0483263, 1.5018],
    [-1.50233, 0.199159, 3.8511],
    [-0.0273012, -4.13347, 0.203112]
    ]))
result = geom_comp_utils.is_valid_SO3(R)
assert not result

print("success")
exit()

dataset_root = '/Users/johnlambert/Downloads/Tanks_and_Temples_Barn_410'
scene_name = 'Barn' # 'Truck'

img_dir = f'{dataset_root}/{scene_name}'
poses_fpath = f'{dataset_root}/{scene_name}_COLMAP_SfM.log'
lidar_ply_fpath = f'{dataset_root}/{scene_name}.ply'
colmap_ply_fpath = f'{dataset_root}/{scene_name}_COLMAP.ply'
ply_alignment_fpath = f'{dataset_root}/{scene_name}_trans.txt'
bounding_polyhedron_json_fpath = f'{dataset_root}/{scene_name}.json'
loader = TanksAndTemplesLoader(
    img_dir=img_dir,
    poses_fpath=poses_fpath,
    lidar_ply_fpath=lidar_ply_fpath,
    ply_alignment_fpath=ply_alignment_fpath,
    bounding_polyhedron_json_fpath=bounding_polyhedron_json_fpath,
    colmap_ply_fpath=colmap_ply_fpath,
)

intrinsics = loader.get_camera_intrinsics_full_res(index=0)

# View frustums
# for index in loader.wTi_gt_dict.keys():
#     wTc = loader.get_camera_pose(index)

wTi_list = [loader.get_camera_pose(index) for index in range(len(loader))]
calibrations = [loader.get_camera_intrinsics_full_res(index) for index in range(len(loader))]

frustums = open3d_vis_utils.create_all_frustums_open3d(
    wTi_list=wTi_list, calibrations=calibrations, frustum_ray_len= 0.3
)
geometries = frustums
lidar_pcd = loader.get_lidar_point_cloud()
colmap_pcd = loader.get_colmap_point_cloud()

open3d.visualization.draw_geometries(geometries + [lidar_pcd] + [colmap_pcd])
exit()

# Compute 2-view error using a front-end.
verifier = LoRansac(use_intrinsics_in_verification=True, estimation_threshold_px=0.5)




import time
start = time.time()
keypoints_list, match_indices_dict = loader.generate_synthetic_correspondences(
    images = [],
    image_pairs = [(i1,i2)]
)
end = time.time()
duration = end - start
print(f"Took {duration} sec.")
keypoints_i1, keypoints_i2 = keypoints_list


i1 = 0
i2 = 1


wTi1 = loader.get_camera_pose(index=i1)
wTi2 = loader.get_camera_pose(index=i2)

i2Ti1 = wTi2.between(wTi1)
i2Ri1_expected = i2Ti1.rotation()
i2Ui1_expected = Unit3(i2Ti1.translation())

camera_intrinsics_i1 = loader.get_camera_intrinsics_full_res(index=i1)
camera_intrinsics_i2 = loader.get_camera_intrinsics_full_res(index=i2)

i2Ri1_computed, i2Ui1_computed, verified_indices_computed, _ = verifier.verify(
    keypoints_i1,
    keypoints_i2,
    match_indices_dict[(i1,i2)],
    camera_intrinsics_i1,
    camera_intrinsics_i2,
)

rot_angular_err = geom_comp_utils.compute_relative_rotation_angle(i2Ri1_expected, i2Ri1_computed)
direction_angular_err = geom_comp_utils.compute_relative_unit_translation_angle(i2Ui1_expected, i2Ui1_computed)

print(f"Errors: rotation {rot_angular_err:.2f}, direction {direction_angular_err:.2f}")


# if i2Ri1_expected is None:
#     self.assertIsNone(i2Ri1_computed)
# else:
#     angular_err = geom_comp_utils.compute_relative_rotation_angle(i2Ri1_expected, i2Ri1_computed)
#     self.assertLess(
#         angular_err,
#         ROTATION_ANGULAR_ERROR_DEG_THRESHOLD,
#         msg=f"Angular error {angular_err:.1f} vs. tol. {ROTATION_ANGULAR_ERROR_DEG_THRESHOLD:.1f}",
#     )
# if i2Ui1_expected is None:
#     self.assertIsNone(i2Ui1_computed)
# else:
#     self.assertLess(
#         geom_comp_utils.compute_relative_unit_translation_angle(i2Ui1_expected, i2Ui1_computed),
#         DIRECTION_ANGULAR_ERROR_DEG_THRESHOLD,
#     )
# np.testing.assert_array_equal(verified_indices_computed, verified_indices_expected)



import gtsfm.utils.io as io_utils

# Generate random image pairs.
# image_pairs = 

# Could enforce that they are roughly on the same side of an object.

# import numpy as np
# # Project LiDAR point cloud into image 1.
# pcd = loader.get_lidar_point_cloud()
# points = np.asarray(pcd.points)

# # Project mesh vertices into image 1.
# mesh = loader.reconstruct_mesh()
# points = np.asarray(mesh.vertices)

# camera_i1 = loader.get_camera(index=0)

# keypoints_i1 = []
# for point in points:
#     keypoints_i1.append(camera_i1.projectSafe(point)[0])

# keypoints_i1 = np.array(keypoints_i1)

# import matplotlib.pyplot as plt
# img = loader.get_image_full_res(index=0)
# plt.imshow(img.value_array.astype(np.uint8))
# plt.scatter(keypoints_i1[:,0], keypoints_i1[:,1], 10, color='r', marker='.', alpha=0.007)
# plt.show()


exit()


# pcd = io_utils.read_point_cloud_from_ply(ply_fpath)


#open3d.visualization.draw_geometries([pcd])

# 
#open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

loader.get_image_full_res(index=0)


geometries = [pcd] # [mesh]
for index in loader.wTi_gt_dict.keys():
    wTc = loader.get_camera_pose(index)
    #import pdb; pdb.set_trace()
    line_sets = open3d_vis_utils.draw_coordinate_frame(wTc=wTc, axis_length=1.0)
    geometries.extend(line_sets)

    if index == 0:
        break

    # if index % 10 == 0:
open3d.visualization.draw_geometries(geometries)



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


"""
conda install -c conda-forge embree=2.17.7

pip install pyembree
print(trimesh.ray.ray_pyembree.error)

In [1]: import pyembree

In [2]: import trimesh

In [3]: m = trimesh.creation.icosphere

In [5]: m = trimesh.creation.icosphere()

# check for `ray_pyembree` instead of `ray_triangle`
In [6]: m.ray
"""
