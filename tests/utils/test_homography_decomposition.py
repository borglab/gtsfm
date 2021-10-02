"""

Test cases come from COLMAP https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix_test.cc
which in turn take values from OpenCV.

Author: John Lambert (Python)
"""

from pathlib import Path
from typing import Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Rot3, PinholeCameraCal3Bundler, Pose3, Unit3

import gtsfm.utils.geometry_comparisons as geom_comp_utils
import gtsfm.utils.homography_decomposition as homography_utils
import gtsfm.utils.plane as plane_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.superpoint import SuperPointDetectorDescriptor
from gtsfm.frontend.matcher.superglue_matcher import SuperGlueMatcher
from gtsfm.frontend.verifier.homography import RansacHomographyEstimator
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.scene_optimizer import FeatureExtractor, TwoViewEstimator
from gtsfm.two_view_estimator import TwoViewEstimationReport


TEST_DATA_ROOT = Path(__file__).parent.parent.resolve() / "data"


def test_decompose_homography_matrix() -> None:
    """Ensure the 4 possible rotation, translation, normal vector options can be computed from H.

    See https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix_test.cc#L44
    # As noted by COLMAP: test case values are obtained from OpenCV.
    """
    H = np.array(
        [
            [2.649157564634028, 4.583875997496426, 70.694447785121326],
            [-1.072756858861583, 3.533262150437228, 1513.656999614321649],
            [0.001303887589576, 0.003042206876298, 1],
        ]
    )
    H *= 3

    # fmt: off
    K = np.array(
        [
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ]
    )
    # fmt: on

    # gtsam.Rot3 returned
    R_cmbs, t_cmbs, n_cmbs = homography_utils.decompose_homography_matrix(H, K1=K, K2=K)

    assert len(R_cmbs) == 4
    assert len(t_cmbs) == 4
    assert len(n_cmbs) == 4

    R_ref = np.array(
        [
            [0.43307983549125, 0.545749113549648, -0.717356090899523],
            [-0.85630229674426, 0.497582023798831, -0.138414255706431],
            [0.281404038139784, 0.67421809131173, 0.682818960388909],
        ]
    )
    t_ref = np.array([1.826751712278038, 1.264718492450820, 0.195080809998819])
    n_ref = np.array([-0.244875830334816, -0.480857890778889, -0.841909446789566])

    ref_solution_exists = False

    for i in range(4):
        if (
            np.allclose(R_cmbs[i].matrix(), R_ref)
            and np.allclose(t_cmbs[i].point3(), t_ref)
            and np.allclose(n_cmbs[i], n_ref)
        ):
            ref_solution_exists = True

    assert ref_solution_exists


def test_pose_from_homography_matrix() -> None:
    """Ensure relative pose is correctly extracted from a homography matrix.

    See: https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix_test.cc#L120
    """
    # defaults to identity 3x3, with focal lengths 1
    camera_intrinsics_i1 = Cal3Bundler()
    camera_intrinsics_i2 = Cal3Bundler()

    R_ref = np.eye(3)
    t_ref = np.array([1, 0, 0], dtype=np.float32)
    n_ref = np.array([-1, 0, 0], dtype=np.float32)
    d_ref = 1  # orthogonal distance to plane
    H = homography_utils.homography_matrix_from_pose(
        camera_intrinsics_i1, camera_intrinsics_i2, R_ref, t_ref, n_ref, d_ref
    )

    # fmt: off
    points1 = np.array(
        [
            [0.1, 0.4],
            [0.2, 0.3],
            [0.3, 0.2],
            [0.4, 0.1]
        ]
    )
    # fmt: on

    points2 = np.zeros((4, 2))
    for i, point1 in enumerate(points1):
        # affine to homogeneous
        point2 = H @ np.array([point1[0], point1[1], 1.0])
        # convert homogenous to affine
        point2 /= point2[2]
        points2[i] = point2[:2]

    R, t, n, points3D = homography_utils.pose_from_homography_matrix(
        H,
        camera_intrinsics_i1,
        camera_intrinsics_i2,
        points1,
        points2,
    )

    np.testing.assert_almost_equal(R.matrix(), R_ref)
    np.testing.assert_almost_equal(t, t_ref)
    np.testing.assert_almost_equal(n, n_ref)
    assert len(points3D) == len(points1)


def test_pose_from_homography_matrix_notre_dame() -> None:
    """Purely planar scene.

    Check SuperPoint + SuperGlue + OpenCV RANSAC-5pt frontend (Essential matrix estimation).

    Essential matrix decomposition on the following image pair yields:
    {
        "i1": 11,
        "i2": 12,
        "i1_filename": "beggs_2603656317.jpg",
        "i2_filename": "beggs_2604036425.jpg",
        "rotation_angular_error": 0.28,
        "translation_angular_error": 72.44,
        "num_inliers_gt_model": 1394,
        "inlier_ratio_gt_model": 1.0,
        "inlier_ratio_est_model": 0.95,
        "num_inliers_est_model": 1394
    }

    """
    images_dir = TEST_DATA_ROOT / "notre-dame" / "images"
    colmap_files_dirpath = TEST_DATA_ROOT / "notre-dame" / "notre-dame-20-colmap"
    loader = ColmapLoader(colmap_files_dirpath=colmap_files_dirpath, images_dir=images_dir, max_frame_lookahead=20)

    assert loader.get_image(0).file_name == "beggs_2603656317.jpg"
    assert loader.get_image(1).file_name == "beggs_2604036425.jpg"

    wTi1 = loader.get_camera_pose(0)
    wTi2 = loader.get_camera_pose(1)

    i2Ti1_gt = wTi2.between(wTi1)

    i2Ri1_dict, i2Ui1_dict, two_view_report_dict = __run_superglue_front_end(loader)

    import pdb; pdb.set_trace()
    # TODO: these thresholds may be impossible
    assert two_view_report_dict[(0,1)].R_error_deg < 5
    assert two_view_report_dict[(0,1)].U_error_deg < 5


"""
Another homography dataset (planar graffiti scene).
https://www.robots.ox.ac.uk/~vgg/data/affine/
"""


def __run_superglue_front_end(loader: LoaderBase) -> TwoViewEstimationReport:
    """Run all image pairs from a data loader through the SuperPoint + SuperGlue front-end."""
    det_desc = SuperPointDetectorDescriptor()
    feature_extractor = FeatureExtractor(det_desc)
    two_view_estimator = TwoViewEstimator(
        matcher=SuperGlueMatcher(use_outdoor_model=True),
        verifier=Ransac(
            use_intrinsics_in_verification=True, estimation_threshold_px=4, min_allowed_inlier_ratio_est_model=0.1
        ),
        eval_threshold_px=4,
        min_num_inliers_acceptance=15,
    )
    i2Ri1_graph_dict, i2Ui1_graph_dict, two_view_report_dict = __get_frontend_computation_graph(
        loader, feature_extractor, two_view_estimator
    )

    with dask.config.set(scheduler="single-threaded"):
        i2Ri1_results, i2Ui1_results, two_view_report_results = dask.compute(
            i2Ri1_graph_dict, i2Ui1_graph_dict, two_view_report_dict
        )
    return i2Ri1_results, i2Ui1_results, two_view_report_results


def __get_frontend_computation_graph(
    loader: LoaderBase, feature_extractor: FeatureExtractor, two_view_estimator: TwoViewEstimator
) -> Tuple[Delayed, Delayed]:
    """Copied from SceneOptimizer class, without back-end code"""
    image_pair_indices = loader.get_valid_pairs()
    image_graph = loader.create_computation_graph_for_images()
    camera_intrinsics_graph = loader.create_computation_graph_for_intrinsics()
    image_shape_graph = loader.create_computation_graph_for_image_shapes()

    gt_pose_graph = loader.create_computation_graph_for_poses()

    # detection and description graph
    keypoints_graph_list = []
    descriptors_graph_list = []
    for delayed_image in image_graph:
        delayed_dets, delayed_descs = feature_extractor.create_computation_graph(delayed_image)
        keypoints_graph_list += [delayed_dets]
        descriptors_graph_list += [delayed_descs]

    # estimate two-view geometry and get indices of verified correspondences.
    i2Ri1_graph_dict = {}
    i2Ui1_graph_dict = {}
    two_view_report_dict = {}
    for (i1, i2) in image_pair_indices:
        gt_i2Ti1 = dask.delayed(lambda x, y: x.between(y))(gt_pose_graph[i2], gt_pose_graph[i1])
        (i2Ri1, i2Ui1, v_corr_idxs, two_view_report) = two_view_estimator.create_computation_graph(
            keypoints_graph_list[i1],
            keypoints_graph_list[i2],
            descriptors_graph_list[i1],
            descriptors_graph_list[i2],
            camera_intrinsics_graph[i1],
            camera_intrinsics_graph[i2],
            image_shape_graph[i1],
            image_shape_graph[i2],
            gt_i2Ti1,
        )
        i2Ri1_graph_dict[(i1, i2)] = i2Ri1
        i2Ui1_graph_dict[(i1, i2)] = i2Ui1
        two_view_report_dict[(i1, i2)] = two_view_report

    return i2Ri1_graph_dict, i2Ui1_graph_dict, two_view_report_dict


def test_pose_from_homography_skydio() -> None:
    """
    311: S1014913.JPG
    324: S1014926.JPG

    11.16 degrees of rotation error, and 156.54 errors of translation error w/o homography consideration
    seems planar.
    """
    # TODO(johnwlambert): add additional test for these planar matches.
    pass


# def test_check_cheirality() -> None:
#     """Ensure that points obviously behind the camera are treated as cheirality failures during triangulation."""

#     # consider i2's frame to be the world frame
#     i2Ri1 = Rot3() # set to identity.
#     i2ti1 = np.array([1,0,0]) # set baseline to 1 meter.

#     cam_i1 = PinholeCameraCal3Bundler()
#     cam_i2 = PinholeCameraCal3Bundler()

#     # halfway between the cameras, and 10 meters ahead
#     point_3d = np.array([0.5,0,10])
#     points1 = cam_i1.projectSafe(point_3d)
#     points2 = cam_i2.projectSafe(point_3d)
#     points3D = check_cheirality(i2Ri1, i2ti1, points1, points2)

#     assert len(points3D) == 0


def test_compute_opposite_of_minor_M00() -> None:
    """Ensure negative of lower-right 2x2 determinant is returned (M00)."""
    # fmt: off
    matrix = np.array(
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
    )
    # fmt: on
    neg_minor = homography_utils.compute_opposite_of_minor(matrix, row=0, col=0)

    # det is -3 = 45 - 48 = 5*9 - 6 * 8
    ref_neg_minor = 3
    assert neg_minor == ref_neg_minor


def test_compute_opposite_of_minor_M22() -> None:
    """Ensure negative of upper-left 2x2 determinant is returned (M22)."""
    # fmt: off
    matrix = np.array(
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
    )
    # fmt: on
    neg_minor = homography_utils.compute_opposite_of_minor(matrix, row=2, col=2)

    # det is -3 = 5 - 8 = 1*5 - 2*4
    ref_neg_minor = 3
    assert neg_minor == ref_neg_minor


def test_compute_opposite_of_minor_M11() -> None:
    """Ensure negative of 2x2 determinant is returned (M11)."""
    # fmt: off
    matrix = np.array(
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
    )
    # fmt: on
    neg_minor = homography_utils.compute_opposite_of_minor(matrix, row=1, col=1)

    # det is -12 = 9 - 21 = 1*9 - 3*7
    ref_neg_minor = 12
    assert neg_minor == ref_neg_minor


def test_pose_from_homography_matrix_planar_scene() -> None:
    """Ensure we can extract pose from homography, for a simple planar scene with 400 points on 1 planes.

    We project the 3d points sampled on a plane to 2d, using known pose.
    """
    np.random.seed(0)
    n_pts = 400 # 10
    # obtain the keypoints and the ground truth essential matrix.
    intrinsics = Cal3Bundler(fx=1000, k1=0, k2=0, u0=1000, v0=500) # suppose (H,W)=(1000,2000).
    keypoints_i1, keypoints_i2, i2Ti1_expected = simulate_planar_scene(N=n_pts, intrinsics=intrinsics)

    # match keypoints row by row
    match_indices = np.hstack([np.arange(n_pts).reshape(-1,1), np.arange(n_pts).reshape(-1,1)])

    # import pdb; pdb.set_trace()

    homography_estimator = RansacHomographyEstimator(estimation_threshold_px=4)
    H, num_H_inliers, H_inlier_ratio, H_inlier_idxs = homography_estimator.estimate(
        keypoints_i1,
        keypoints_i2,
        match_indices=match_indices
    )
    h_corr_idxs = match_indices[H_inlier_idxs]
    i2Ri1, i2Ui1, _, _ = homography_utils.pose_from_homography_matrix(
        H=H,
        camera_intrinsics_i1=intrinsics,
        camera_intrinsics_i2=intrinsics,
        points1=keypoints_i1.coordinates[h_corr_idxs[:, 0]],
        points2=keypoints_i2.coordinates[h_corr_idxs[:, 1]]
    )

    rot_angular_error = geom_comp_utils.compute_relative_rotation_angle(i2Ti1_expected.rotation(), i2Ri1)
    assert np.isclose(rot_angular_error, 0, atol=1e-3)

    # given wti1 = np.array([0, -1, -5])
    # given wti2 = np.array([2, 0, -5.4])
    # expected translation i2ti1 = (-2, -1, 0.44)
    translation_error_vec = Unit3(i2Ti1_expected.translation()).point3() - i2Ui1.point3()
    assert np.allclose(translation_error_vec, 0, atol=1e-3)
    import pdb; pdb.set_trace()


def simulate_planar_scene(N: int, intrinsics: Cal3Bundler) -> Tuple[Keypoints, Keypoints, Pose3]:
    """Generate a scene where 3D points are on one plane, and projects the points to the 2 cameras.

    There are N points on plane 1.

    Camera 1 is 1 meter above Camera 2 (in -y direction).
    Camera 2 is 0.4 meters behind Camera 1 (in -z direction).

       cam 1                        plane @ z=10
       o ----                         |
       |           |                  |
       |         --|-- +z             |
                   | world origin     |
    o -----
    |
    | cam 2

    Args:
        N: number of points on plane.
        intrinsics: intrinsics for both cameras.

    Returns:
        keypoints for image i1, of length (N).
        keypoints for image i2, of length (N).
        Relative pose i2Ti1.
    """
    # range of 3D points
    range_x_coordinate = (-7, 7)
    range_y_coordinate = (-10, 10)

    # define the plane equation
    # plane at z=10, so ax + by + cz + d = 0 + 0 + -z + 10 = 0
    plane1_coeffs = (0, 0, -1, 10)

    # sample the points from planes
    points_3d = plane_utils.sample_points_on_plane(plane1_coeffs, range_x_coordinate, range_y_coordinate, N)

    # import visualization.open3d_vis_utils as open3d_vis_utils
    # colors = np.zeros_like(points_3d).astype(np.uint8)
    # colors[:, 0] = 255
    # spheres = open3d_vis_utils.create_colored_spheres_open3d(
    #     point_cloud=points_3d, rgb=colors, sphere_radius=0.1
    # )

    # define the camera poses and compute the essential matrix
    wti1 = np.array([0, -1, -5])
    wti2 = np.array([2, 0, -5.4])

    wRi1 = Rot3.RzRyRx(x=0.0, y=np.deg2rad(1), z=0.0)
    wRi2 = Rot3.RzRyRx(x=0.0, y=np.deg2rad(-1), z=0.0)

    wTi1 = Pose3(wRi1, wti1)
    wTi2 = Pose3(wRi2, wti2)
    i2Ti1 = wTi2.between(wTi1)

    # project 3D points to 2D image measurements
    camera_i1 = PinholeCameraCal3Bundler(wTi1, intrinsics)
    camera_i2 = PinholeCameraCal3Bundler(wTi2, intrinsics)

    # coord_frame = open3d_vis_utils.draw_coordinate_frame(wTc=Pose3(), axis_length= 1.0)
    # frustums = open3d_vis_utils.create_all_frustums_open3d([wTi1, wTi2], [intrinsics]*2)
    # import open3d
    # open3d.visualization.draw_geometries(spheres + frustums + coord_frame)

    uv_im1 = []
    uv_im2 = []
    for point in points_3d:
        uv_im1.append(camera_i1.project(point))
        uv_im2.append(camera_i2.project(point))

    uv_im1 = np.vstack(uv_im1)
    uv_im2 = np.vstack(uv_im2)

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import pdb; pdb.set_trace()
    plt.scatter(uv_im1[:,0], uv_im1[:,1], 1, color='r', marker='.')
    plt.title("im1 keypoints")
    plt.show()

    plt.scatter(uv_im2[:,0], uv_im2[:,1], 1, color='r', marker='.')
    plt.title("im2 keypoints")
    plt.show()


    # return the points as keypoints and the relative pose
    return Keypoints(coordinates=uv_im1), Keypoints(coordinates=uv_im2), i2Ti1


if __name__ == "__main__":

    # test_pose_from_homography_matrix_planar_scene()
    test_pose_from_homography_matrix_notre_dame()
