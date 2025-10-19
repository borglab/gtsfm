"""Unit test for the DataAssociation class (and implicitly the Point3dInitializer class).

Triangulation examples from:
     borglab/gtsam/python/gtsam/tests/test_Triangulation.py
     gtsam/geometry/tests/testTriangulation.cpp

Authors: Sushmita Warrier, Xiaolong Wu, John Lambert
"""

import unittest
from typing import Dict, List, Tuple

import dask
import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point2Vector, Point3, Pose3, Pose3Vector, Rot3  # type: ignore
from gtsam.utils.test_case import GtsamTestCase  # type: ignore

from gtsfm.common.keypoints import Keypoints
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.data_association.dsf_tracks_estimator import DsfTracksEstimator
from gtsfm.data_association.point3d_initializer import TriangulationOptions, TriangulationSamplingMode
from gtsfm.products.visibility_graph import AnnotatedGraph, ImageIndexPairs


def get_pose3_vector(num_poses: int) -> Pose3Vector:
    """Generate camera poses for use in triangulation tests"""

    # Looking along X-axis, 1 meter above ground plane (x-y)
    upright = Rot3.Ypr(-np.pi / 2, 0.0, -np.pi / 2)
    pose1 = Pose3(upright, Point3(0, 0, 1))

    # create second camera 1 meter to the right of first camera
    pose2 = pose1.compose(Pose3(Rot3(), Point3(1, 0, 0)))

    # Add third camera slightly rotated
    rotatedCamera = Rot3.Ypr(0.1, 0.2, 0.1)
    pose3 = pose1.compose(Pose3(rotatedCamera, Point3(0.1, -2, -0.1)))

    available_poses = [pose1, pose2, pose3]

    pose3_vec = Pose3Vector()
    for i in range(num_poses):
        pose3_vec.append(available_poses[i])
    return pose3_vec


def generate_noisy_2d_measurements(
    world_point: np.ndarray, calibrations: List[Cal3Bundler], per_image_noise_vecs: np.ndarray, poses: Pose3Vector
) -> Tuple[List[Keypoints], ImageIndexPairs, Dict[int, PinholeCameraCal3Bundler]]:
    """
    Generate PinholeCameras from specified poses and calibrations, and then generate
    1 measurement per camera of a given 3d point.

    Args:
        world_point: 3d coords of 3d landmark in world frame
        calibrations: List of calibrations for each camera
        noise_params: List of amounts of noise to be added to each measurement
        poses: List of poses for each camera in world frame

    Returns:
        keypoints_list: List of keypoints in all images (projected measurements in all images)
        img_idxs: Tuple of indices for all images
        cameras: Dictionary mapping image index i to calibrated PinholeCamera object
    """
    keypoints_list = []
    measurements = Point2Vector()
    cameras = dict()
    for i in range(len(poses)):
        camera = PinholeCameraCal3Bundler(poses[i], calibrations[i])
        # Project landmark into two cameras and triangulate
        z = camera.project(world_point)
        cameras[i] = camera
        measurement = z + per_image_noise_vecs[i]
        measurements.append(measurement)
        keypoints_list += [Keypoints(coordinates=measurement.reshape(1, 2))]

    # Create image indices for each pose - only subsequent pairwise matches
    # assumed, e.g. between images (0,1) and images (1,2)
    img_idxs = []
    for i in range(len(poses) - 1):
        img_idxs += [(i, i + 1)]

    return keypoints_list, img_idxs, cameras


def get_2d_tracks(correspondences: AnnotatedGraph[np.ndarray], keypoints_list: List[Keypoints]) -> List[SfmTrack2d]:
    tracks_estimator = DsfTracksEstimator()
    return tracks_estimator.run(correspondences, keypoints_list)


class TestDataAssociation(GtsamTestCase):
    """Unit tests for data association module, which maps the feature tracks to their 3D landmarks."""

    def setUp(self):
        """Set up the data association module."""
        super().setUp()

        # landmark ~5 meters infront of camera
        self.expected_landmark = Point3(5, 0.5, 1.2)

        # shared calibration
        f, k1, k2, u0, v0 = 1500, 0, 0, 640, 480
        self.sharedCal = Cal3Bundler(f, k1, k2, u0, v0)

    def test_ransac_sample_biased_baseline_sharedCal_2poses(self):
        """ """
        mode = TriangulationSamplingMode.RANSAC_SAMPLE_BIASED_BASELINE
        self.verify_triangulation_sharedCal_2poses(mode)

    def test_ransac_topk_baselines_sharedCal_2poses(self):
        """ """
        mode = TriangulationSamplingMode.RANSAC_TOPK_BASELINES
        self.verify_triangulation_sharedCal_2poses(mode)

    def test_ransac_sample_uniform_sharedCal_2poses(self):
        """ """
        mode = TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM
        self.verify_triangulation_sharedCal_2poses(mode)

    def test_no_ransac_sharedCal_2poses(self):
        """ """
        mode = TriangulationSamplingMode.NO_RANSAC
        self.verify_triangulation_sharedCal_2poses(mode)

    def verify_triangulation_sharedCal_2poses(self, triangulation_mode: TriangulationSamplingMode):
        """Tests that the triangulation is accurate for shared calibration with a specified triangulation mode.

        Checks whether the triangulated landmark map formed from 2 measurements is valid, if min track length = 3
        (should be invalid)

        The noise vectors represent the amount of noise to be added to measurements.
        """
        keypoints_list, _, cameras = generate_noisy_2d_measurements(
            world_point=self.expected_landmark,
            calibrations=[self.sharedCal, self.sharedCal],
            per_image_noise_vecs=np.array([[-0.1, -0.5], [0.2, -0.3]]),
            poses=get_pose3_vector(num_poses=2),
        )

        # create matches
        # since there is only one measurement in each image, both assigned feature index 0
        matches_dict = {(0, 1): np.array([[0, 0]])}

        triangulation_options = TriangulationOptions(
            reproj_error_threshold=5, mode=triangulation_mode, min_num_hypotheses=20
        )
        tracks_2d = get_2d_tracks(matches_dict, keypoints_list)
        da = DataAssociation(min_track_len=3, triangulation_options=triangulation_options)
        sfm_tracks, avg_track_reproj_errors, triangulation_exit_codes = da.run_triangulation(
            cameras=cameras, tracks_2d=tracks_2d
        )
        triangulated_landmark_map, _ = da.assemble_gtsfm_data_from_tracks(
            num_images=len(cameras),
            cameras=cameras,
            tracks_2d=tracks_2d,
            sfm_tracks=sfm_tracks,
            avg_track_reproj_errors=avg_track_reproj_errors,
            triangulation_exit_codes=triangulation_exit_codes,
            cameras_gt=[None] * len(cameras),
            relative_pose_priors={},
        )
        # assert that we cannot obtain even 1 length-3 track if we have only 2 camera poses
        # result should be empty, since nb_measurements < min track length
        assert (
            triangulated_landmark_map.number_tracks() == 0
        ), "Failure: tracks exceed expected track length (should be 0 tracks)"

    def test_triangulation_individualCal_without_ransac(self):
        """Tests that the triangulation is accurate for individual camera calibration, without RANSAC-based
        triangulation. Checks if cameras and triangulated 3D point are as expected.
        """
        k1 = 0
        k2 = 0
        f, u0, v0 = 1500, 640, 480
        f_, u0_, v0_ = 1600, 650, 440
        K1 = Cal3Bundler(f, k1, k2, u0, v0)
        K2 = Cal3Bundler(f_, k1, k2, u0_, v0_)

        keypoints_list, _, cameras = generate_noisy_2d_measurements(
            world_point=self.expected_landmark,
            calibrations=[K1, K2],
            per_image_noise_vecs=np.zeros((2, 2)),
            poses=get_pose3_vector(num_poses=2),
        )

        # create matches
        # since there is only one measurement in each image, both assigned feature index 0
        matches_dict = {(0, 1): np.array([[0, 0]])}

        triangulation_options = TriangulationOptions(reproj_error_threshold=5, mode=TriangulationSamplingMode.NO_RANSAC)
        da = DataAssociation(min_track_len=2, triangulation_options=triangulation_options)

        tracks_2d = get_2d_tracks(matches_dict, keypoints_list)
        sfm_tracks, avg_track_reproj_errors, triangulation_exit_codes = da.run_triangulation(
            cameras=cameras, tracks_2d=tracks_2d
        )
        sfm_data, _ = da.assemble_gtsfm_data_from_tracks(
            num_images=len(cameras),
            cameras=cameras,
            tracks_2d=tracks_2d,
            sfm_tracks=sfm_tracks,
            avg_track_reproj_errors=avg_track_reproj_errors,
            triangulation_exit_codes=triangulation_exit_codes,
            cameras_gt=[None] * len(cameras),
            relative_pose_priors={},
        )
        estimated_landmark = sfm_data.get_track(0).point3()
        self.gtsamAssertEquals(estimated_landmark, self.expected_landmark, 1e-2)

        for i in sfm_data.get_valid_camera_indices():
            self.gtsamAssertEquals(sfm_data.get_camera(i), cameras.get(i))

    def test_ransac_sample_biased_baseline_sharedCal_3poses(self):
        """ """
        mode = TriangulationSamplingMode.RANSAC_SAMPLE_BIASED_BASELINE
        self.verify_triangulation_sharedCal_3poses(mode)

    def test_ransac_topk_baselines_sharedCal_3poses(self):
        """ """
        mode = TriangulationSamplingMode.RANSAC_TOPK_BASELINES
        self.verify_triangulation_sharedCal_3poses(mode)

    def test_ransac_sample_uniform_sharedCal_3poses(self):
        """ """
        mode = TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM
        self.verify_triangulation_sharedCal_3poses(mode)

    def test_no_ransac_sharedCal_3poses(self):
        """ """
        mode = TriangulationSamplingMode.NO_RANSAC
        self.verify_triangulation_sharedCal_3poses(mode)

    def verify_triangulation_sharedCal_3poses(self, triangulation_mode: TriangulationSamplingMode):
        """Tests that the triangulation is accurate for shared calibration with a specified triangulation mode.

        Checks whether the sfm data formed from 3 measurements is valid. The noise vectors represent the amount of
        noise to be added to measurements.
        """
        keypoints_list, _, cameras = generate_noisy_2d_measurements(
            world_point=self.expected_landmark,
            calibrations=[self.sharedCal, self.sharedCal, self.sharedCal],
            per_image_noise_vecs=np.array([[-0.1, -0.5], [-0.2, 0.3], [0.1, -0.1]]),
            poses=get_pose3_vector(num_poses=3),
        )

        # create matches
        # since there is only one measurement in each image, both assigned feature index 0
        matches_dict = {(0, 1): np.array([[0, 0]]), (1, 2): np.array([[0, 0]])}

        triangulation_options = TriangulationOptions(
            reproj_error_threshold=5, mode=triangulation_mode, min_num_hypotheses=20
        )
        da = DataAssociation(min_track_len=3, triangulation_options=triangulation_options)

        tracks_2d = get_2d_tracks(matches_dict, keypoints_list)
        sfm_tracks, avg_track_reproj_errors, triangulation_exit_codes = da.run_triangulation(
            cameras=cameras, tracks_2d=tracks_2d
        )
        sfm_data, _ = da.assemble_gtsfm_data_from_tracks(
            num_images=len(cameras),
            cameras=cameras,
            tracks_2d=tracks_2d,
            sfm_tracks=sfm_tracks,
            avg_track_reproj_errors=avg_track_reproj_errors,
            triangulation_exit_codes=triangulation_exit_codes,
            cameras_gt=[None] * len(cameras),
            relative_pose_priors={},
        )

        estimated_landmark = sfm_data.get_track(0).point3()
        # checks if computed 3D point is as expected
        self.gtsamAssertEquals(estimated_landmark, self.expected_landmark, 1e-2)

        # checks if number of tracks are as expected, should be just 1, over all 3 cameras
        assert sfm_data.number_tracks() == 1, "more tracks than expected"
        # checks if cameras saved to result are as expected
        for i in cameras.keys():
            self.gtsamAssertEquals(sfm_data.get_camera(i), cameras.get(i))

    def test_data_association_with_missing_camera(self):
        """Tests the data association with input tracks which use a camera index for which the camera doesn't exist."""

        triangulation_options = TriangulationOptions(
            reproj_error_threshold=5, mode=TriangulationSamplingMode.NO_RANSAC, min_num_hypotheses=20
        )
        da = DataAssociation(min_track_len=3, triangulation_options=triangulation_options)

        # add cameras 0 and 2
        cameras = {
            0: PinholeCameraCal3Bundler(Pose3(Rot3.RzRyRx(0, np.deg2rad(20), 0), np.zeros((3, 1)))),
            2: PinholeCameraCal3Bundler(Pose3(Rot3.RzRyRx(0, 0, 0), np.array([10, 0, 0]))),
        }

        # just have one track, chaining cams 0->1 , and cams 1->2
        correspondences = {(0, 1): np.array([[0, 0]], dtype=np.int32), (1, 2): np.array([[0, 0]], dtype=np.int32)}
        keypoints_shared = Keypoints(coordinates=np.array([[20.0, 10.0]]))

        # will lead to a cheirality exception because keypoints are identical in two cameras
        # no track will be formed, and thus connected component will be empty
        tracks_2d = get_2d_tracks(correspondences, [keypoints_shared] * 3)
        sfm_tracks, avg_track_reproj_errors, triangulation_exit_codes = da.run_triangulation(
            cameras=cameras, tracks_2d=tracks_2d
        )
        sfm_data, _ = da.assemble_gtsfm_data_from_tracks(
            num_images=3,
            cameras=cameras,
            tracks_2d=tracks_2d,
            sfm_tracks=sfm_tracks,
            avg_track_reproj_errors=avg_track_reproj_errors,
            triangulation_exit_codes=triangulation_exit_codes,
            cameras_gt=[None] * 3,
            relative_pose_priors={},
        )

        self.assertEqual(len(sfm_data.get_valid_camera_indices()), 0)
        self.assertEqual(sfm_data.number_tracks(), 0)

    def test_create_computation_graph(self):
        """Tests the graph to create data association for 3 images. Checks if result from dask computation graph is the
        same as result without dask."""
        keypoints_list, img_idxs, cameras = generate_noisy_2d_measurements(
            world_point=self.expected_landmark,
            calibrations=[self.sharedCal, self.sharedCal, self.sharedCal],
            per_image_noise_vecs=np.array([[-0.1, -0.5], [-0.2, 0.3], [0.1, -0.1]]),
            poses=get_pose3_vector(num_poses=3),
        )

        cameras_gt = [None] * len(cameras)

        # create matches
        # since there is only one measurement in each image, both assigned feature index 0
        corr_idxs_graph = {(0, 1): np.array([[0, 0]]), (1, 2): np.array([[0, 0]])}
        tracks_2d = get_2d_tracks(corr_idxs_graph, keypoints_list)

        # Run without computation graph
        triangulation_options = TriangulationOptions(
            reproj_error_threshold=5, mode=TriangulationSamplingMode.RANSAC_TOPK_BASELINES, min_num_hypotheses=20
        )
        da = DataAssociation(min_track_len=3, triangulation_options=triangulation_options)

        # Run without delayed computation graph.
        expected_sfm_data, _ = da.run_triangulation_and_evaluate(
            num_images=len(cameras),
            cameras=cameras,
            tracks_2d=tracks_2d,
            cameras_gt=cameras_gt,
            relative_pose_priors={},
        )

        # Run with delayed computation graph.
        delayed_sfm_data = da.create_computation_graph(
            num_images=len(cameras),
            cameras=cameras,
            tracks_2d=tracks_2d,
            cameras_gt=cameras_gt,
            relative_pose_priors=dask.delayed({}),
        )

        with dask.config.set(scheduler="single-threaded"):
            (dask_sfm_data,) = dask.compute(delayed_sfm_data)

        assert expected_sfm_data.number_tracks() == dask_sfm_data.number_tracks(), "Dask not configured correctly"

        for k in range(expected_sfm_data.number_tracks()):
            assert (
                expected_sfm_data.get_track(k).numberMeasurements() == dask_sfm_data.get_track(k).numberMeasurements()
            ), "Dask tracks incorrect"
            # Test if the measurement in both are equal
            np.testing.assert_array_almost_equal(
                expected_sfm_data.get_track(k).measurement(0)[1],
                dask_sfm_data.get_track(k).measurement(0)[1],
                decimal=1,
                err_msg="Dask measurements incorrect",
            )
        for i in expected_sfm_data.get_valid_camera_indices():
            self.gtsamAssertEquals(expected_sfm_data.get_camera(i), cameras.get(i))


if __name__ == "__main__":
    unittest.main()
