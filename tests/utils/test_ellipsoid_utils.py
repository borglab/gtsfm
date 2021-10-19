"""Unit tests for functions in ellipsoid utils file.

Authors: Adi Singh
"""
from typing import Dict, Tuple
import unittest
from gtsfm.common.gtsfm_data import GtsfmData
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, SfmTrack, Similarity3

import numpy as np
import numpy.testing as npt

import gtsfm.utils.ellipsoid as ellipsoid_utils
from scipy.io import loadmat


class TestEllipsoidUtils(unittest.TestCase):
    """Class containing all unit tests for ellipsoid utils."""

    def test_get_ortho_axis_alignment_transform(self) -> None:
        """Tests the get_ortho_axis_alignment_transform() function with a GtsfmData object containing 3 camera frustums
        and 6 points in the point cloud. All points lie on z=0 plane. All frustums lie on z=2 plane and look down on
        the z=0 plane.

           sample_data:              output_data:

               y                         y
               |     c                   |
           o   |                         o
             c | c                       |
          ------------- x   ==>  --o--c-----c--o-- x
             o | c                       |
               |   o                     c
               |                         |
               |                         |
               |                         c

        c = point at (xi,yi,0) with a camera frustum at (xi,yi,2)
        o = point at (xi,yi,0)
        """

        sample_data = GtsfmData(number_images=3)
        default_intrinsics = Cal3Bundler(fx=100, k1=0, k2=0, u0=0, v0=0)

        # Add 3 camera frustums to sample_data (looking down at z=0 plane)
        camera0 = PinholeCameraCal3Bundler(Pose3(Rot3(np.eye(3)), np.array([-1, 1, 2])), default_intrinsics)
        camera1 = PinholeCameraCal3Bundler(Pose3(Rot3(np.eye(3)), np.array([1, 1, 2])), default_intrinsics)
        camera2 = PinholeCameraCal3Bundler(Pose3(Rot3(np.eye(3)), np.array([1, -1, 2])), default_intrinsics)

        sample_data.add_camera(0, camera0)
        sample_data.add_camera(1, camera1)
        sample_data.add_camera(2, camera2)

        # Add 6 tracks to sample_data

        # fmt: off
        points3d = np.array(
            [
                [1, 1, 0],
                [-1, 1, 0],
                [-2, 2, 0],
                [-1, -1, 0],
                [1, -1, 0],
                [2, -2, 0],
                [5, 5, 0]
            ]
        )
        # fmt: on

        for pt_3d in points3d:
            sample_data.add_track(SfmTrack(pt_3d))

        # Apply alignment transformation to sample_data
        walignedTw = ellipsoid_utils.get_ortho_axis_alignment_transform(sample_data)
        walignedTw = Similarity3(R=walignedTw.rotation(), t=walignedTw.translation(), s=1.0)
        sample_data = sample_data.apply_Sim3(walignedTw)

        # Verify correct 3d points.
        computed_3d_points = np.array([sample_data.get_track(i).point3() for i in range(sample_data.number_tracks())])
        expected_3d_points = np.array(
            [
                [0, -np.sqrt(2), 0],
                [np.sqrt(2), 0, 0],
                [2 * np.sqrt(2), 0, 0],
                [0, np.sqrt(2), 0],
                [-np.sqrt(2), 0, 0],
                [-2 * np.sqrt(2), 0, 0],
                [0, -5 * np.sqrt(2), 0],
            ]
        )
        npt.assert_almost_equal(computed_3d_points, expected_3d_points, decimal=6)

        # Verify correct camera poses.
        expected_wTi_list = [
            Pose3(walignedTw.rotation(), np.array([np.sqrt(2), 0, 2])),
            Pose3(walignedTw.rotation(), np.array([0, -np.sqrt(2), 2])),
            Pose3(walignedTw.rotation(), np.array([-np.sqrt(2), 0, 2])),
        ]

        computed_wTi_list = sample_data.get_camera_poses()
        for wTi_computed, wTi_expected in zip(computed_wTi_list, expected_wTi_list):
            assert wTi_computed.equals(wTi_expected, tol=1e-9)

    def test_point_cloud_cameras_locked(self) -> None:
        """Tests the get_ortho_axis_alignment_transform() function with a GtsfmData object containing 11 point cloud
        points and 12 camera frustums from the door-12 dataset. Determines if the points are properly "locked" in
        with one another before and after the alignment transformation is applied.
        """

        sample_data = GtsfmData(number_images=12)

        # Read all 12 camera poses from data.mat file which contains extrinsics for all cameras in door12 dataset.
        data = loadmat("data.mat")
        M_list = [data["P"][0][i] for i in range(12)]
        K = M_list[0][:3, :3]
        Kinv = np.linalg.inv(K)
        iTw_list = [Kinv @ M_list[i] for i in range(12)]
        wTi_list = [Pose3(Rot3(iTw[:3, :3]), iTw[:, 3]).inverse() for iTw in iTw_list]

        # Add 12 camera frustums to sample_data.
        default_intrinsics = Cal3Bundler(fx=100, k1=0, k2=0, u0=0, v0=0)
        for idx, pose in enumerate(wTi_list):
            camera = PinholeCameraCal3Bundler(pose, default_intrinsics)
            sample_data.add_camera(idx, camera)

        # fmt: off
        points_3d = np.array(
            [
                [-1.4687794397729077, -1.4966178675020756, 14.583277665978546],
                [-1.6172612359102505, -1.0951470733744013, 14.579095414379562],
                [-3.2190882723771783, -4.463465966172758, 14.444076631000476],
                [-0.6754206497590093, -1.1132530165104157, 14.916222213341355],
                [-1.5514099044537981, -1.305810425894855, 14.584788688422206],
                [-1.551319353347404, -1.304881682597853, 14.58246449772602],
                [-1.9055918588057448, -1.192867982227922, 14.446379510423219],
                [-1.5936792439193013, -1.4398818807488012, 14.587749795933021],
                [-1.5937405395983737, -1.4401641027442411, 14.588167699143174],
                [-1.6599318889904735, -1.2273604755959784, 14.57861988411431],
                [2.1935589900444867, 1.6233406628428935, 12.610234497076608]
            ]
        )
        # fmt: on

        # Add all point cloud points to sample_data
        for point_3d in points_3d:
            sample_data.add_track(SfmTrack(point_3d))

        camera_translations = np.array([pose.translation() for pose in sample_data.get_camera_poses()])
        initial_relative_distances = self.compute_relative_distances(camera_translations, points_3d)

        # Apply alignment transformation to sample_data
        walignedTw = ellipsoid_utils.get_ortho_axis_alignment_transform(sample_data)
        walignedTw = Similarity3(R=walignedTw.rotation(), t=walignedTw.translation(), s=1.0)
        sample_data = sample_data.apply_Sim3(walignedTw)

        # Aggregate the final, transformed points
        num_tracks = sample_data.number_tracks()
        transformed_points_3d = [np.array(sample_data.get_track(i).point3()) for i in range(num_tracks)]
        transformed_points_3d = np.array(transformed_points_3d)
        transformed_camera_translations = np.array([pose.translation() for pose in sample_data.get_camera_poses()])

        final_relative_distances = self.compute_relative_distances(
            transformed_camera_translations, transformed_points_3d
        )

        npt.assert_almost_equal(final_relative_distances, initial_relative_distances, decimal=6)

    def compute_relative_distances(
        self, camera_translations: np.ndarray, points_3d: np.ndarray
    ) -> Dict[Tuple[int, int], float]:
        """Computes the relative distances between every camera frustum and every point in the point cloud.
        Let M be the number of cameras and N be the number of point cloud points.

        Args:
            camera_translations: camera center coordinates, shape (M,3).
            points_3d: points in the point cloud, shape (N,3).

        Returns:
            Array containing relative distances between each camera center to every point in the point cloud,
            shape (M, N).

        Raises:
            TypeError: if collection of points is not of shape (N,3).
        """
        if camera_translations.shape[1] != 3 or points_3d.shape[1] != 3:
            raise TypeError("Points should be 3 dimensional")

        M = camera_translations.shape[0]
        N = points_3d.shape[0]

        relative_distances = np.zeros((M, N))

        for camera_index in range(0, M):
            for point_index in range(0, N):
                camera_center = camera_translations[camera_index, :]
                point = points_3d[point_index, :]

                relative_distances[camera_index, point_index] = np.linalg.norm(point - camera_center)

        return relative_distances

    def test_center_point_cloud(self) -> None:
        """Tests the center_point_cloud() function with 3 sample points.

        Means of x,y,z is clearly (2,2,2), so centering the point cloud yields:
            (1,1,1) -> (-1,-1,-1)
            (2,2,2) -> (0,0,0)
            (3,3,3) -> (1,1,1)
        """

        sample_points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        computed, mean = ellipsoid_utils.center_point_cloud(sample_points)
        expected = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        npt.assert_almost_equal(computed, expected, decimal=6)

    def test_center_point_cloud_wrong_dims(self) -> None:
        """Tests the center_point_cloud() function with 5 sample points of 2 dimensions."""

        sample_points = np.array([[6, 13], [5, 9], [6, 10]])
        self.assertRaises(TypeError, ellipsoid_utils.center_point_cloud, sample_points)

    def test_remove_outlier_points(self) -> None:
        """Tests the remove_outlier_points() function with 5 sample points."""

        # fmt: off
        sample_points = np.array(
            [
                [0.5, 0.6, 0.8],
                [0.9, 1, 0.2],
                [20, 20, 20],  # the outlier point to be filtered
                [0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3],
            ]
        )
        # fmt: on

        computed = ellipsoid_utils.remove_outlier_points(sample_points)
        expected = np.array([[0.5, 0.6, 0.8], [0.9, 1, 0.2], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
        npt.assert_almost_equal(computed, expected, decimal=6)

    def test_remove_outlier_points_wrong_dims(self) -> None:
        """Tests the remove_outlier_points() function with 5 sample points of 2 dimensions."""

        sample_points = np.array([[6, 13], [5, 9], [6, 10], [9, 13], [5, 12]])
        self.assertRaises(TypeError, ellipsoid_utils.remove_outlier_points, sample_points)

    def test_get_alignment_rotation_matrix_from_svd(self) -> None:
        """Tests the get_alignment_rotation_matrix_from_svd() function with 6 sample points. Transforms a rotated cross
        to points lying on the x and y axes.

        sample_points:        aligned_points:

             |                       |
         o   |                       o
           o | o                     |
        -------------   ==>  --o--o-----o--o--
           o | o                     |
             |   o                   o
             |                       |

         o = point
        """

        # fmt: off
        sample_points = np.array(
            [
                [1, 1, 0],
                [-1, 1, 0],
                [-2, 2, 0],
                [-1, -1, 0],
                [1, -1, 0],
                [2, -2, 0]
            ]
        )
        # fmt: on

        computed_rotation = ellipsoid_utils.get_alignment_rotation_matrix_from_svd(sample_points)
        num = np.sqrt(2) / 2
        expected_rotation = np.array([[-num, num, 0], [-num, -num, 0], [0, 0, 1]])
        npt.assert_almost_equal(computed_rotation, expected_rotation, decimal=6)

        # Apply the rotation transformation to sample_points
        # Verify that every aligned point's x or y coordinate is 0
        aligned_points = sample_points @ computed_rotation.T

        closeToZero = np.isclose(aligned_points[:, :2], 0)
        assert np.all(closeToZero[:, 0] | closeToZero[:, 1])

    def test_get_alignment_rotation_matrix_from_svd_wrong_dims(self) -> None:
        """Tests the get_alignment_rotation_matrix_from_svd() function with 6 sample points of 2 dimensions."""

        sample_points = np.array([[1, 1], [-1, 1], [-2, 2], [-1, -1], [1, -1], [2, -2]])
        self.assertRaises(TypeError, ellipsoid_utils.get_alignment_rotation_matrix_from_svd, sample_points)


if __name__ == "__main__":
    unittest.main()
