"""
Unit tests on homography verification/estimation.

Author: John Lambert
"""

from typing import Tuple

import numpy as np
from gtsam import Cal3Bundler, Rot3, PinholeCameraCal3Bundler, Pose3

import gtsfm.utils.sampling as sampling_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.homography_verifier.ransac import RansacHomographyVerifier


def test_verify_homography_inliers_minimalset() -> None:
    """Fit homography on minimal set of 4 correspondences, w/ no outliers. Homography provides a 2x scaling.

    Image 1:       Image 2:
                       X       X
       |               |
       o   o           |
       |               |
    ---o---o-      ----X-------X
       |               |
    """
    # fmt: off
    uv_i1 = np.array(
    	[
    		[0,0],
    		[1,0],
    		[1,1],
    		[0,1]
    	]
    )
    uv_i2 = np.array(
    	[
    		[0,0],
    		[2,0],
    		[2,2],
    		[0,2]
    	]
    )
    # fmt: on
    keypoints_i1 = Keypoints(coordinates=uv_i1)
    keypoints_i2 = Keypoints(coordinates=uv_i2)
    # fmt: off
    match_indices = np.array(
    	[
    		[0,0],
    		[1,1],
    		[2,2],
    		[3,3]
    	]
    )

    # fmt: on
    estimator = RansacHomographyVerifier()
    H, H_inlier_idxs, inlier_ratio, num_inliers = estimator.verify(
        keypoints_i1, keypoints_i2, match_indices, estimation_threshold_px=4
    )

    # fmt: off
    expected_H = np.array(
        [
            [2,  0, 0],
            [0,  2, 0],
            [0,  0, 1]
        ]
    )
    # fmt: on
    assert np.allclose(H, expected_H)

    expected_H_inlier_idxs = np.array([0, 1, 2, 3])
    assert np.allclose(H_inlier_idxs, expected_H_inlier_idxs)

    assert inlier_ratio == 1.0
    assert num_inliers == 4


# def test_estimate_homography_inliers_corrupted() -> None:
#     """Fit homography on set of 6 correspondences, w/ 2 outliers."""

#     # fmt: off
#     uv_i1 = np.array(
#       [
#           [0,0],
#           [1,0],
#           [1,1],
#           [0,1],
#           [0,1000], # outlier
#           [0,2000] # outlier
#       ]
#     )

#     # # 2x multiplier on uv_i1
#     # uv_i2 = np.array(
#     #     [
#     #         [0,0],
#     #         [2,0],
#     #         [2,2],
#     #         [0,2],
#     #         [500,0], # outlier
#     #         [1000,0] # outlier
#     #     ]
#     # )

#  #    # fmt: on
#   # keypoints_i1 = Keypoints(coordinates=uv_i1)
#   # keypoints_i2 = Keypoints(coordinates=uv_i2)
#   # # fmt: off
#   # match_indices = np.array(
#   #   [
#   #       [0,0],
#   #       [1,1],
#   #       [2,2],
#   #       [3,3],
#   #       [4,4],
#   #       [5,5]
#   #   ]
#   # )

#   # fmt: on
#   estimator = RansacHomographyEstimator()
#   num_inliers, inlier_ratio = estimator.estimate(
#       keypoints_i1,
#       keypoints_i2,
#       match_indices
#   )

#   assert inlier_ratio == 4/6
#   assert num_inliers == 4



def test_verify_homography_planar_geometry() -> None:
    """Generate 400 points on a single 3d plane, project them to 2d, and fit a homography to them."""
    np.random.seed(0)
    n_pts = 400  # 10
    # obtain the keypoints and the ground truth essential matrix.
    intrinsics = Cal3Bundler(fx=1000, k1=0, k2=0, u0=1000, v0=500)  # suppose (H,W)=(1000,2000).
    keypoints_i1, keypoints_i2, i2Ti1_expected = simulate_planar_scene(N=n_pts, intrinsics=intrinsics)

    # match keypoints row by row
    match_indices = np.hstack([np.arange(n_pts).reshape(-1, 1), np.arange(n_pts).reshape(-1, 1)])

    homography_estimator = RansacHomographyVerifier()
    H, H_inlier_idxs, inlier_ratio, num_inliers = homography_estimator.verify(
        keypoints_i1, keypoints_i2, match_indices=match_indices, estimation_threshold_px=4
    )

    assert isinstance(H, np.ndarray)
    assert H.shape == (3,3)

    expected_H_inlier_idxs = np.arange(n_pts)
    assert np.allclose(H_inlier_idxs, expected_H_inlier_idxs)

    assert inlier_ratio == 1.0
    assert num_inliers == n_pts


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
        intrinsics: shared intrinsics for both cameras.

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
    points_3d = sampling_utils.sample_points_on_plane(plane1_coeffs, range_x_coordinate, range_y_coordinate, N)

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

    uv_im1 = []
    uv_im2 = []
    for point in points_3d:
        uv_im1.append(camera_i1.project(point))
        uv_im2.append(camera_i2.project(point))

    uv_im1 = np.vstack(uv_im1)
    uv_im2 = np.vstack(uv_im2)

    # return the points as keypoints and the relative pose
    return Keypoints(coordinates=uv_im1), Keypoints(coordinates=uv_im2), i2Ti1

