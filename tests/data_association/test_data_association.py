"""Unit test for the DataAssociation class.

Authors: Sushmita Warrier
"""
import unittest
from typing import Dict, List, Tuple

import dask
import gtsam
import numpy as np
from common.keypoints import Keypoints
from gtsam import (
    Cal3Bundler,
    PinholeCameraCal3Bundler,
    Point2Vector,
    Point3,
    Pose3,
    Pose3Vector,
    Rot3,
)
from gtsam.utils.test_case import GtsamTestCase

from data_association.data_assoc import DataAssociation, TriangulationParam
from data_association.feature_tracks import FeatureTrackGenerator


class TestDataAssociation(GtsamTestCase):
    """
    Unit tests for data association module, which maps the feature tracks to their 3D landmarks.
    """

    def setUp(self):
        """
        Set up the data association module.
        """
        super().setUp()

        # set up ground truth data for comparison

        self.dummy_matches = {
            (0, 1): np.array([[0, 2]]),
            (1, 2): np.array([[2, 3], [4, 5], [7, 9]]),
            (0, 2): np.array([[1, 8]]),
        }
        img1_kp_coords = np.array([[12, 16], [13, 18], [0, 10]])
        img1_kp_scale = np.array([6.0, 9.0, 8.5])
        img2_kp_coords = np.array(
            [
                [8, 2],
                [16, 14],
                [22, 23],
                [1, 6],
                [50, 50],
                [16, 12],
                [82, 121],
                [39, 60],
            ]
        )
        img3_kp_coords = np.array(
            [
                [1, 1],
                [8, 13],
                [40, 6],
                [82, 21],
                [1, 6],
                [12, 18],
                [15, 14],
                [25, 28],
                [7, 10],
                [14, 17],
            ]
        )
        self.feature_list = [
            Keypoints(coordinates=img1_kp_coords, scale=img1_kp_scale),
            Keypoints(coordinates=img2_kp_coords),
            Keypoints(coordinates=img3_kp_coords),
        ]
        self.malformed_matches = {
            (0, 1): np.array([[0, 2]]),
            (1, 2): np.array([[2, 3], [4, 5], [7, 9]]),
            (0, 2): np.array([[1, 8]]),
            (1, 1): np.array([[0, 3]]),
        }

        # Generate two poses for use in triangulation tests
        # Looking along X-axis, 1 meter above ground plane (x-y)
        upright = Rot3.Ypr(-np.pi / 2, 0.0, -np.pi / 2)
        pose1 = Pose3(upright, Point3(0, 0, 1))

        # create second camera 1 meter to the right of first camera
        pose2 = pose1.compose(Pose3(Rot3(), Point3(1, 0, 0)))

        self.poses = Pose3Vector()
        self.poses.append(pose1)
        self.poses.append(pose2)

        # landmark ~5 meters infront of camera
        self.expected_landmark = Point3(5, 0.5, 1.2)

        # shared calibration
        f, k1, k2, u0, v0 = 1500, 0, 0, 640, 480
        self.sharedCal = Cal3Bundler(f, k1, k2, u0, v0)

    def test_track(self) -> None:
        """
        Tests that the tracks are being merged and mapped correctly from the dummy matches.
        """
        self.track = FeatureTrackGenerator(self.dummy_matches, self.feature_list)
        # len(track) value for toy case strictly
        assert len(self.track.filtered_landmark_data) == 4, "tracks incorrectly mapped"

    def test_primary_filtering(self) -> None:
        """
        Tests that the tracks are being filtered correctly.
        Removes tracks that have two measurements in a single image.
        """
        filtered_map = FeatureTrackGenerator(
            self.malformed_matches, self.feature_list
        ).filtered_landmark_data

        # check that the length of the observation list corresponding to each key
        # is the same. Only good tracks will remain
        assert len(filtered_map) == 4, "Tracks not filtered correctly"

    def test_triangulation_sharedCal_without_ransac(self) -> None:
        """
        Tests that the triangulation is accurate for shared calibration.
        Example from borglab/gtsam/python/gtsam/tests/test_Triangulation.py
        and gtsam/geometry/tests/testTriangulation.cpp
        1. Checks whether the triangulated landmark map formed from 2 measurements is valid, if min track length = 3.
        2. Checks whether the triangulated landmark map formed from 3 measurements is valid.
            Also checks if the triangulated point is as expected.
        """

        matches_1, feature_list, poses, _, cameras = self.__generate_2_poses(
            self.sharedCal
        )

        # hyperparameters for unit test
        reproj_error_thresh = 5  # pixels
        min_track_len = 3  # at least 3 measurements required

        da = DataAssociation(reproj_error_thresh, min_track_len)
        triangulated_landmark_map = da.run(matches_1, feature_list, cameras)
        # result should be empty, since nb_measurements < min track length
        assert (
            triangulated_landmark_map.number_tracks() == 0
        ), "tracks exceeding expected track length"

        # compute triangulated map for 3 measurements
        matches_2, feature_list, cameras = self.__generate_3_poses(self.sharedCal)
        da = DataAssociation(reproj_error_thresh, min_track_len)
        triangulated_landmark_map = da.run(matches_2, feature_list, cameras)
        computed_landmark = triangulated_landmark_map.track(0).point3()

        # checks if number of tracks are as expected
        assert (
            triangulated_landmark_map.number_tracks() == 1
        ), "more tracks than expected"
        # checks if cameras saved to result are as expected
        for cam in range(triangulated_landmark_map.number_cameras()):
            self.gtsamAssertEquals(
                triangulated_landmark_map.camera(cam), cameras.get(cam)
            )
        # checks if computed 3D point is as expected
        self.gtsamAssertEquals(computed_landmark, self.expected_landmark, 1e-2)

    def test_triangulation_sharedCal_ransac_uniform(self) -> None:
        """
        Tests that the triangulation is accurate for shared calibration.
        Example from borglab/gtsam/python/gtsam/tests/test_Triangulation.py
        and gtsam/geometry/tests/testTriangulation.cpp
        """

        matches_1, feature_list, poses, _, cameras = self.__generate_2_poses(
            self.sharedCal
        )

        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=3,  # at least 3 measurements required
            mode=TriangulationParam.RANSAC_SAMPLE_UNIFORM,
            num_hypotheses=20,
        )
        triangulated_landmark_map = da.run(matches_1, feature_list, cameras)

        assert (
            triangulated_landmark_map.number_tracks() == 0
        ), "tracks exceeding expected track length"

        matches_2, feature_list, cameras = self.__generate_3_poses(self.sharedCal)
        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=3,  # at least 3 measurements required
            mode=TriangulationParam.RANSAC_SAMPLE_UNIFORM,
            num_hypotheses=20,
        )
        triangulated_landmark_map = da.run(matches_2, feature_list, cameras)
        computed_landmark = triangulated_landmark_map.track(0).point3()
        assert (
            triangulated_landmark_map.number_tracks() == 1
        ), "more tracks than expected"
        self.gtsamAssertEquals(computed_landmark, self.expected_landmark, 1e-2)
        for cam in range(triangulated_landmark_map.number_cameras()):
            self.gtsamAssertEquals(
                triangulated_landmark_map.camera(cam), cameras.get(cam)
            )

    def test_triangulation_sharedCal_ransac_baseline(self):
        """
        Tests that the triangulation is accurate for shared calibration.
        Example from borglab/gtsam/python/gtsam/tests/test_Triangulation.py
        and gtsam/geometry/tests/testTriangulation.cpp
        """
        matches_1, feature_list, _, _, cameras = self.__generate_2_poses(self.sharedCal)
        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=3,  # at least 3 measurements required
            mode=TriangulationParam.RANSAC_SAMPLE_BIASED_BASELINE,
            num_hypotheses=20,
        )
        triangulated_landmark_map = da.run(matches_1, feature_list, cameras)

        assert (
            triangulated_landmark_map.number_tracks() == 0
        ), "tracks exceeding expected track length"

        matches_2, feature_list, cameras = self.__generate_3_poses(self.sharedCal)
        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=3,  # at least 3 measurements required
            mode=TriangulationParam.RANSAC_SAMPLE_BIASED_BASELINE,
            num_hypotheses=20,
        )
        triangulated_landmark_map = da.run(matches_2, feature_list, cameras)
        computed_landmark = triangulated_landmark_map.track(0).point3()
        assert (
            triangulated_landmark_map.number_tracks() == 1
        ), "more tracks than expected"

        for cam in range(triangulated_landmark_map.number_cameras()):
            self.gtsamAssertEquals(
                triangulated_landmark_map.camera(cam), cameras.get(cam)
            )

        self.gtsamAssertEquals(computed_landmark, self.expected_landmark, 1e-2)

    def test_triangulation_sharedCal_ransac_maxtomin(self):
        """
        Tests that the triangulation is accurate for shared calibration.
        Example from borglab/gtsam/python/gtsam/tests/test_Triangulation.py
        and gtsam/geometry/tests/testTriangulation.cpp
        """
        matches_1, feature_list, poses, _, cameras = self.__generate_2_poses(
            self.sharedCal
        )
        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=3,  # at least 3 measurements required
            mode=TriangulationParam.RANSAC_TOPK_BASELINES,
            num_hypotheses=20,
        )
        triangulated_landmark_map = da.run(matches_1, feature_list, cameras)

        assert (
            triangulated_landmark_map.number_tracks() == 0
        ), "tracks exceeding expected track length"

        matches_2, feature_list, cameras = self.__generate_3_poses(self.sharedCal)
        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=3,  # at least 3 measurements required
            mode=TriangulationParam.RANSAC_TOPK_BASELINES,
            num_hypotheses=20,
        )
        triangulated_landmark_map = da.run(matches_2, feature_list, cameras)
        computed_landmark = triangulated_landmark_map.track(0).point3()
        assert (
            triangulated_landmark_map.number_tracks() == 1
        ), "more tracks than expected"
        self.gtsamAssertEquals(computed_landmark, self.expected_landmark, 1e-2)

        for cam in range(triangulated_landmark_map.number_cameras()):
            self.gtsamAssertEquals(
                triangulated_landmark_map.camera(cam), cameras.get(cam)
            )

    def test_triangulation_individualCal_without_ransac(self):
        """
        Tests that the triangulation is accurate for individual camera calibration, without RANSAC-based triangulation.
        Checks if cameras and triangulated 3D point saved to result are as expected.
        """
        f, k1, k2, u0, v0 = 1500, 0, 0, 640, 480
        f_2, k1_2, k2_2, u0_2, v0_2 = 1600, 0, 0, 650, 440
        K1 = Cal3Bundler(f, k1, k2, u0, v0)
        K2 = Cal3Bundler(f_2, k1_2, k2_2, u0_2, v0_2)

        measurements, feature_list, img_idxs, cameras = self.__generate_measurements(
            (K1, K2), (0.0, 0.0), self.poses
        )

        # since there is only one measurement in each image, both assigned feature index 0
        matched_idxs = np.array([[0, 0]])

        # create matches
        matches = {img_idxs: matched_idxs}

        da = DataAssociation(5, 2)
        triangulated_landmark_map = da.run(matches, feature_list, cameras)
        computed_landmark = triangulated_landmark_map.track(0).point3()
        self.gtsamAssertEquals(computed_landmark, self.expected_landmark, 1e-2)
        for cam in range(triangulated_landmark_map.number_cameras()):
            self.gtsamAssertEquals(
                triangulated_landmark_map.camera(cam), cameras.get(cam)
            )

    def test_create_computation_graph(self):
        """
        Tests the graph to create data association for 3 images.
        Checks if result from dask computation graph is the same as result without dask.
        """
        matches, features, cameras = self.__generate_3_poses(self.sharedCal)

        # Run without computation graph
        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=3,  # at least 3 measurements required
            mode=TriangulationParam.RANSAC_TOPK_BASELINES,
            num_hypotheses=20,
        )
        expected_landmark_map = da.run(matches, features, cameras)

        # Run with computation graph
        computed_landmark_map = da.create_computation_graph(matches, features, cameras)

        with dask.config.set(scheduler="single-threaded"):
            dask_result = dask.compute(computed_landmark_map)[0]

        assert (
            expected_landmark_map.number_tracks() == dask_result.number_tracks()
        ), "Dask not configured correctly"

        for i in range(expected_landmark_map.number_tracks()):
            assert (
                expected_landmark_map.track(i).number_measurements()
                == dask_result.track(i).number_measurements()
            ), "Dask tracks incorrect"
            # Test if the measurement in both are equal
            np.testing.assert_array_almost_equal(
                expected_landmark_map.track(i).measurement(0)[1],
                dask_result.track(i).measurement(0)[1],
                1,
                "Dask measurements incorrect",
            )
        for cam in range(expected_landmark_map.number_cameras()):
            self.gtsamAssertEquals(expected_landmark_map.camera(cam), cameras.get(cam))

    def __generate_2_poses(
        self, sharedCal: Cal3Bundler
    ) -> Tuple[
        Dict[Tuple[int], np.array],
        List[Keypoints],
        Pose3Vector,
        Point2Vector,
        Dict[int, PinholeCameraCal3Bundler],
    ]:
        """
        Generate 2 matches and their corresponding poses for shared calibration.

        Args:
            sharedCal: shared calibration for cameras

        Returns:
            matches: dictionary, with key as image pair (i1,i2) and value
                            as matching keypoint indices.
            feature_list: List of keypoints
            poses: List of poses
            measurements: List of projected measurements in all images
            cameras: List of cameras
        """
        # Amount of noise to be added to measurements
        noise_params = (-np.array([0.1, 0.5]), -np.array([-0.2, 0.3]))

        measurements, feature_list, img_idxs, cameras = self.__generate_measurements(
            (sharedCal, sharedCal), noise_params, self.poses
        )

        # since there is only one measurement in each image, both assigned feature index 0
        matched_idxs = np.array([[0, 0]])

        # create matches
        matches_1 = {img_idxs: matched_idxs}
        return matches_1, feature_list, self.poses, measurements, cameras

    def __generate_3_poses(
        self, sharedCal: Cal3Bundler
    ) -> Tuple[
        Dict[Tuple[int], np.array], List[Keypoints], Dict[int, PinholeCameraCal3Bundler]
    ]:
        """
        Generate 3 matches and corresponding poses with shared calibration.

        Args:
            sharedCal: shared calibration for cameras

        Returns:
            matches: dictionary, with key as image pair (i1,i2) and value
                            as matching keypoint indices.
            feature_list: List of keypoints
            cameras: List of cameras
        """
        matches, feature_list, poses, measurements, cameras = self.__generate_2_poses(
            sharedCal
        )

        # Add third camera slightly rotated
        rotatedCamera = Rot3.Ypr(0.1, 0.2, 0.1)
        pose3 = Pose3(rotatedCamera, Point3(0.1, -2, -0.1))
        camera3 = PinholeCameraCal3Bundler(pose3, sharedCal)
        # Camera_idx hardcoded here for 3rd camera
        cameras.update({2: camera3})
        z3 = camera3.project(self.expected_landmark)
        # add noise to measurement
        measurements.append(z3 + np.array([0.1, -0.1]))
        poses.append(pose3)

        img_idxs2 = tuple(list(range(1, len(poses))))
        feature_list.append(Keypoints(coordinates=np.array([measurements[2]])))

        # Only one measurement in images 1 and 2, hence each get index 0
        matched_idxs2 = np.array([[0, 0]])
        matches.update({img_idxs2: matched_idxs2})
        return matches, feature_list, cameras

    def __generate_measurements(
        self,
        calibration: Tuple[Cal3Bundler, Cal3Bundler],
        noise_params: Tuple[np.ndarray, np.ndarray],
        poses: Pose3Vector,
    ) -> Tuple[
        Point2Vector, List[Keypoints], Tuple[int], Dict[int, PinholeCameraCal3Bundler]
    ]:
        """
        Generate measurements, features, img indices and cameras for given calibration and poses

        Args:
            calibration: Tuple of calibrations for each camera
            noise_params: Tuple of amounts of noise to be added to each     measurement
            poses: List of poses for each camera

        Returns:
            measurements: List of projected measurements in all images
            feature_list: List of keypoints in all images
            img_idxs: Tuple of indices for all images
            cameras: List of cameras for all images
        """
        measurements = Point2Vector()
        cameras = dict()
        for i in range(len(poses)):
            camera = PinholeCameraCal3Bundler(poses[i], calibration[i])
            # Project landmark into two cameras and triangulate
            z = camera.project(self.expected_landmark)
            cameras.update({i: camera})
            measurements.append(z + noise_params[i])
        # Create image indices for each pose
        img_idxs = tuple(list(range(len(self.poses))))
        # List of features in each image
        for i in range(len(measurements)):
            feature_list = [Keypoints(coordinates=np.array([m])) for m in measurements]
        return measurements, feature_list, img_idxs, cameras


if __name__ == "__main__":
    unittest.main()
