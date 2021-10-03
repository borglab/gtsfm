"""Unit tests for functions in ellipsoid utils file.

Authors: Adi Singh
"""
import unittest
from gtsfm.common.gtsfm_data import GtsfmData
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, SfmTrack, Similarity3
from gtsfm.utils.ellipsoid import get_ortho_axis_alignment_transform

import numpy as np
import numpy.testing as npt

import gtsfm.utils.ellipsoid as ellipsoid_utils


class TestEllipsoidUtils(unittest.TestCase):
    """Class containing all unit tests for ellipsoid utils."""

    def test_get_ortho_axis_alignment_transform(self) -> None:
        """Tests the get_otho_axis_alignment_transform() function with a GtsfmData object containing 3 camera frustums
        and 6 points in the point cloud. All points lie on z=0 plane. All frustums lie on z=2 plane and look down on
        the z=0 plane.

           sample_data:              output_data:

               y                         y
               |                         |
           o   |                         o
             c | c                       |
          ------------- x   ==>  --o--c-----c--o-- x
             o | c                       |
               |   o                     c
               |                         |

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
        walignedTw = get_ortho_axis_alignment_transform(sample_data)
        sim3_walignedTw = Similarity3(R=walignedTw.rotation(), t=walignedTw.translation(), s=1.0)
        sample_data = sample_data.apply_Sim3(sim3_walignedTw)

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
