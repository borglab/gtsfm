"""Unit tests for functions in ellipsoid utils file.

Authors: Adi Singh
"""
from typing import Dict, Tuple
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
        walignedTw = get_ortho_axis_alignment_transform(sample_data)
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

        # fmt: off
        camera_rotations = np.array(
            [
                [0.9999998447593831, 1.1417402647582947e-05, -0.0005560104530481775, 3.468759506231243e-05],
                [0.9995136351774726, 0.001828651147351114, -0.03008309804543026, 0.008009765370116636],
                [0.9970874902147401, 0.018252605499050226, -0.07262460477616217, 0.014458424218177786],
                [0.9936952358054102, 0.014015743345306576, -0.10906395003410695, 0.02187217591065432],
                [0.989329890103814, -0.0010530184030079614, -0.142189654679541, 0.03173896346025128],
                [0.9833849860715913, -0.0014030690202723683, -0.17862108789671424, 0.03234976854469622],
                [0.9774833274265902, -0.00378574949023753, -0.20836133309086646, 0.033129557444009895],
                [0.9704083254046114, -0.010793591066242695, -0.23833905440090336, 0.03722466285401185],
                [0.960600358539938, -0.012504596462849266, -0.274638208916899, 0.04079755437720905],
                [0.953733866739889, -0.00860199538166799, -0.2955277491681979, 0.05459914450256389],
                [0.9493214986564442, -0.014695628398273269, -0.31101388206318636, 0.04293129230117615],
                [0.9399839877935723, -0.012027558186297983, -0.3359345325083303, 0.0585954810900816]
            ]
        )

        camera_translations = np.array(
            [
                [0.008800121152611574, 0.0007141263181849019, 0.002817403334051625],
                [0.8300887912277424, 0.01410277633103259, -0.06030205404378542],
                [1.7306949813194408, 0.021258565393368827, -0.18892067817191444],
                [2.6910414872549824, 0.07414836494892013, -0.07548834659649717],
                [3.6941764033802267, 0.09544270661110316, 0.01676263485115892],
                [4.673843209495996, 0.07539611837657942, 0.23840610902768944],
                [5.402559601223315, 0.026358515716331174, 0.4586402355996675],
                [6.31333070240205, 0.022871515711057763, 0.6857086248655833],
                [6.923886545179155, 0.06808805928475041, 1.1473408960868587],
                [7.530518308816488, 0.28808544158440896, 1.4819172717072404],
                [8.028592490458774, 0.08865620551543868, 1.7179844667998094],
                [8.476132976532003, 0.32905651920076096, 2.1255119615808105]
            ]
        )
        # fmt: on

        # Add 12 camera frustums to sample_data
        default_intrinsics = Cal3Bundler(fx=100, k1=0, k2=0, u0=0, v0=0)
        for idx, (rotation, translation) in enumerate(zip(camera_rotations, camera_translations)):
            qw, qx, qy, qz = rotation
            camera = PinholeCameraCal3Bundler(Pose3(Rot3(qw, qx, qy, qz), translation), default_intrinsics)
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

        # Add 11 point cloud points to sample_data
        for point_3d in points_3d:
            sample_data.add_track(SfmTrack(point_3d))

        initial_points_3d = np.concatenate((camera_translations, points_3d), axis=0)  # (23, 3)
        initial_relative_distances = self.compute_relative_distances(initial_points_3d)

        # Apply alignment transformation to sample_data
        walignedTw = get_ortho_axis_alignment_transform(sample_data)
        walignedTw = Similarity3(R=walignedTw.rotation(), t=walignedTw.translation(), s=1.0)
        sample_data = sample_data.apply_Sim3(walignedTw)

        # Aggregate the final, transformed points
        num_tracks = sample_data.number_tracks()
        transformed_points_3d = [np.array(sample_data.get_track(i).point3()) for i in range(num_tracks)]
        transformed_points_3d = np.array(transformed_points_3d)
        transformed_camera_translations = np.array([pose.translation() for pose in sample_data.get_camera_poses()])
        transformed_points_3d = np.concatenate(
            (transformed_camera_translations, transformed_points_3d), axis=0
        )  # shape (23,3)

        final_relative_distances = self.compute_relative_distances(transformed_points_3d)

        # Determine if all relative distances remain the same after alignment transformation
        for index_pair in initial_relative_distances.keys():
            initial_distance = initial_relative_distances[index_pair]
            final_distance = final_relative_distances[index_pair]

            npt.assert_almost_equal(initial_distance, final_distance, decimal=6)

    def compute_relative_distances(self, all_points_3d: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Computes the relative distances between all possible pairs of points in all_points_3d and returns a
        dictionary containing all the relative distance information.

        Args:
            all_points_3d: collection of points, shape (N,3).

        Returns:
            Dictionary storing the relative distances between all possible pairs of points.

        Raises:
            TypeError: if collection of points is not of shape (N,3).
        """
        if all_points_3d.shape[1] != 3:
            raise TypeError("Points should be 3 dimensional")

        relative_distances = {}
        N = all_points_3d.shape[0]

        for i in range(0, N):
            for j in range(i + 1, N):
                relative_distances[(i, j)] = np.linalg.norm(all_points_3d[i, :] - all_points_3d[j, :])

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
