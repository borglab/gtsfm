"""Unit tests for functions in ellipsoid utils file.

Authors: Adi Singh
"""

import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import scipy.spatial.distance
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, SfmTrack, Similarity3

import gtsfm.utils.ellipsoid as ellipsoid_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.olsson_loader import OlssonLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"


class TestEllipsoidUtils(unittest.TestCase):
    """Class containing all unit tests for ellipsoid utils."""

    def setUp(self) -> None:
        self.loader = OlssonLoader(str(DATA_ROOT_PATH / "set1_lund_door"))
        assert len(self.loader)

    def test_get_ortho_axis_alignment_transform(self) -> None:
        """Tests the get_ortho_axis_alignment_transform() function with a GtsfmData object containing 3 camera frustums
        and 6 points in the point cloud. All points lie on z=0 plane. All frustums lie on z=2 plane and look down on
        the z=0 plane.

           sample_data:              output_data:

               y                         y
               |                         o
               |                         |
               |     o                   |
           o   |                         c
             c | c                       |
          ------------- x   ==>  --o--c-----c--o-- x
             o | c                       |
               |   o                     o
               |                         |

        c = point at (xi,yi,0) with a camera frustum at (xi,yi,2)
        o = point at (xi,yi,0)
        """
        sample_data = GtsfmData(number_images=3)
        default_intrinsics = Cal3Bundler(fx=100, k1=0, k2=0, u0=0, v0=0)

        # Add 3 camera frustums to sample_data (looking down at z=0 plane)
        cam_translations = np.array([[-1, 1, 2], [1, 1, 2], [1, -1, 2]])

        for i in range(len(cam_translations)):
            camera = PinholeCameraCal3Bundler(Pose3(Rot3(), cam_translations[i, :]), default_intrinsics)
            sample_data.add_camera(i, camera)

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
                [5, 5, 0]   # represents an outlier in this set of points
            ]
        )
        # fmt: on

        for pt_3d in points3d:
            sample_data.add_track(SfmTrack(pt_3d))

        # Apply alignment transformation to sample_data
        walignedTw = ellipsoid_utils.get_ortho_axis_alignment_transform(sample_data)
        walignedSw = Similarity3(R=walignedTw.rotation(), t=walignedTw.translation(), s=1.0)
        sample_data = sample_data.transform_with_sim3(walignedSw)

        # Verify correct 3d points.
        computed_3d_points = np.array([sample_data.get_track(i).point3() for i in range(sample_data.number_tracks())])
        expected_3d_points = np.array(
            [
                [0, np.sqrt(2), 0],
                [-np.sqrt(2), 0, 0],
                [-2 * np.sqrt(2), 0, 0],
                [0, -np.sqrt(2), 0],
                [np.sqrt(2), 0, 0],
                [2 * np.sqrt(2), 0, 0],
                [0, 5 * np.sqrt(2), 0],
            ]
        )
        npt.assert_almost_equal(computed_3d_points, expected_3d_points, decimal=3)

        # Verify correct camera poses.
        expected_wTi_list = [
            Pose3(walignedTw.rotation(), np.array([-np.sqrt(2), 0, 2])),
            Pose3(walignedTw.rotation(), np.array([0, np.sqrt(2), 2])),
            Pose3(walignedTw.rotation(), np.array([np.sqrt(2), 0, 2])),
        ]

        computed_wTi_list = sample_data.get_camera_poses_list()
        assert len(computed_wTi_list) == len(expected_wTi_list)
        for wTi_computed, wTi_expected in zip(computed_wTi_list, expected_wTi_list):
            assert wTi_computed.equals(wTi_expected, tol=1e-9)

    def test_point_cloud_cameras_locked(self) -> None:
        """Tests the get_ortho_axis_alignment_transform() function with a GtsfmData object containing 11 point cloud
        points and 12 camera frustums from the door-12 dataset. Determines if the points and frustums are properly
        "locked" in with one another before and after the alignment transformation is applied.
        """
        sample_data = GtsfmData(number_images=12)

        # Instantiate OlssonLoader to read camera poses from door12 dataset.
        wTi_list = self.loader._wTi_list

        # Add 12 camera frustums to sample_data.
        default_intrinsics = Cal3Bundler(fx=100, k1=0, k2=0, u0=0, v0=0)
        for idx, pose in enumerate(wTi_list):
            camera = PinholeCameraCal3Bundler(pose, default_intrinsics)
            sample_data.add_camera(idx, camera)

        # fmt: off
        # These points are taken directly from the first 11 points generated by GTSFM on the door12 dataset (without
        # any separate alignment transformation being applied)
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

        camera_translations = np.array([pose.translation() for pose in sample_data.get_camera_poses_list()])
        initial_relative_distances = scipy.spatial.distance.cdist(camera_translations, points_3d, metric="euclidean")

        # Apply alignment transformation to sample_data
        walignedTw = ellipsoid_utils.get_ortho_axis_alignment_transform(sample_data)
        walignedSw = Similarity3(R=walignedTw.rotation(), t=walignedTw.translation(), s=1.0)
        sample_data = sample_data.transform_with_sim3(walignedSw)

        # Aggregate the final, transformed points
        num_tracks = sample_data.number_tracks()
        transformed_points_3d = [np.array(sample_data.get_track(i).point3()) for i in range(num_tracks)]
        transformed_points_3d = np.array(transformed_points_3d)
        transformed_camera_translations = np.array([pose.translation() for pose in sample_data.get_camera_poses_list()])

        final_relative_distances = scipy.spatial.distance.cdist(
            transformed_camera_translations, transformed_points_3d, metric="euclidean"
        )

        npt.assert_almost_equal(final_relative_distances, initial_relative_distances, decimal=3)

    def test_center_point_cloud(self) -> None:
        """Tests the center_point_cloud() function with 3 sample points.

        Means of x,y,z is clearly (2,2,2), so centering the point cloud yields:
            (1,1,1) -> (-1,-1,-1)
            (2,2,2) -> (0,0,0)
            (3,3,3) -> (1,1,1)
        """
        sample_points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        computed = ellipsoid_utils.center_point_cloud(sample_points)
        expected = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        npt.assert_almost_equal(computed, expected, decimal=3)

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

        computed, _ = ellipsoid_utils.remove_outlier_points(sample_points)
        expected = np.array([[0.5, 0.6, 0.8], [0.9, 1, 0.2], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
        npt.assert_almost_equal(computed, expected, decimal=3)

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
        expected_rotation = np.array([[num, -num, 0], [num, num, 0], [0, 0, 1]])
        npt.assert_almost_equal(computed_rotation, expected_rotation, decimal=3)

        # Apply the rotation transformation to sample_points
        # Verify that every aligned point's x or y coordinate is 0
        aligned_points = sample_points @ computed_rotation.T

        closeToZero = np.isclose(aligned_points[:, :2], 0)
        assert np.all(closeToZero[:, 0] | closeToZero[:, 1])

    def test_get_alignment_rotation_matrix_from_svd_wrong_dims(self) -> None:
        """Tests the get_alignment_rotation_matrix_from_svd() function with 6 sample points of 2 dimensions."""

        sample_points = np.array([[1, 1], [-1, 1], [-2, 2], [-1, -1], [1, -1], [2, -2]])
        self.assertRaises(TypeError, ellipsoid_utils.get_alignment_rotation_matrix_from_svd, sample_points)

    def test_get_right_singular_vectors(self) -> None:
        """Tests the get_right_singular_vectors() function by checking that it outputs the same V matrix as
        np.linalg.svd()."""

        # fmt: off
        points = np.array(
            [
                [3,4,5],
                [4,1,3],
                [9,1,2],
                [6,3,1]
            ]
        )
        # fmt: on

        V, singular_values = ellipsoid_utils.get_right_singular_vectors(points)
        computed_Vt = V.T

        U, S, Vt = np.linalg.svd(points, full_matrices=False)
        expected_Vt = Vt

        # assert np.allclose(singular_values, S)

        computed_Vt = np.round(computed_Vt, 3)
        expected_Vt = np.round(expected_Vt, 3)

        # Check if each eigenvector of computed_Vt and expected_Vt has the same direction.
        for rowIdx in range(computed_Vt.shape[0]):
            assert np.all((computed_Vt[rowIdx, :] == expected_Vt[rowIdx, :])) or np.all(
                (computed_Vt[rowIdx, :] == -1 * expected_Vt[rowIdx, :])
            )


if __name__ == "__main__":
    unittest.main()
