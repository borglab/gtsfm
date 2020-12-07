"""Unit test for the DataAssociation class.

Authors: Sushmita Warrier
"""
import unittest

import dask
import numpy as np
import gtsam
from gtsam.utils.test_case import GtsamTestCase
from typing import List, Dict

from data_association.data_assoc import DataAssociation
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

        self.dummy_matches = {(0,1): np.array([[0,2]]), 
                    (1,2): np.array([[2,3], 
                                    [4,5], 
                                    [7,9]]),
                    (0,2): np.array([[1,8]])}
        self.feature_list = [
                        [(12,16, 6), (13,18, 9), (0,10, 8.5)], 
                        [(8,2), (16,14), (22,23), (1,6), (50,50), (16,12), (82,121), (39,60)], 
                        [(1,1), (8,13), (40,6), (82,21), (1,6), (12,18), (15,14), (25,28), (7,10), (14,17)]
                        ]
        self.malformed_matches = {(0,1): np.array([[0,2]]), 
                    (1,2): np.array([[2,3], 
                                    [4,5], 
                                    [7,9]]),
                    (0,2): np.array([[1,8]]),
                    (1,1): np.array([[0,3]])}

        # Generate two poses for use in triangulation tests
        # Looking along X-axis, 1 meter above ground plane (x-y)
        upright = gtsam.Rot3.Ypr(-np.pi / 2, 0., -np.pi / 2)
        pose1 = gtsam.Pose3(upright, gtsam.Point3(0, 0, 1))

        # create second camera 1 meter to the right of first camera
        pose2 = pose1.compose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0)))

        self.poses = gtsam.Pose3Vector()
        self.poses.append(pose1)
        self.poses.append(pose2)

        # landmark ~5 meters infront of camera
        self.expected_landmark = gtsam.Point3(5, 0.5, 1.2)
    
    def test_track(self):
        """
        Tests that the tracks are being merged and mapped correctly
        """
        self.track = FeatureTrackGenerator(self.dummy_matches, self.feature_list)
        # len(track) value for toy case strictly
        assert len(self.track.filtered_landmark_data) == 4, "tracks incorrectly mapped"

    
    def test_primary_filtering(self):
        """
        Tests that the tracks are being filtered correctly.
        Removes tracks that have two measurements in a single image.
        """
        filtered_map = FeatureTrackGenerator(self.malformed_matches, self.feature_list).filtered_landmark_data

        # check that the length of the observation list corresponding to each key is the same. Only good tracks will remain
        assert len(filtered_map) == 4, "Tracks not filtered correctly"

    def test_triangulation_sharedCal(self):
        """
        Tests that the triangulation is accurate for shared calibration. 
        Example from borglab/gtsam/python/gtsam/tests/test_Triangulation.py
        and gtsam/geometry/tests/testTriangulation.cpp
        """  
        sharedCal = gtsam.Cal3Bundler(1500, 0, 0, 640, 480)

        matches_1, feature_list, poses, _ = self.__generate_2_poses(sharedCal)
        da = DataAssociation(matches_1, feature_list)
        triangulated_landmark_map = da.run(poses, True, 5,3, False, sharedCal, None)
        assert len(triangulated_landmark_map) == 0, "tracks exceeding expected track length"
        

        matches_2, feature_list, poses = self.__generate_3_poses(sharedCal)
        da = DataAssociation(matches_2, feature_list)
        triangulated_landmark_map = da.run(poses, True,5,3, False, sharedCal, None)
        computed_landmark = triangulated_landmark_map[0].point3()
        assert len(triangulated_landmark_map)== 1, "more tracks than expected"
        self.gtsamAssertEquals(computed_landmark, self.expected_landmark,1e-2)
    
    def test_triangulation_individualCal(self):
        """
        Tests that the triangulation is accurate for individual camera calibration.
        """
        K1 = gtsam.Cal3Bundler(1500, 0, 0, 640, 480)
        K2 = gtsam.Cal3Bundler(1600, 0, 0, 650, 440)

        measurements, feature_list, img_idxs, cameras = self.__generate_measurements((K1, K2), (0.0, 0.0), self.poses)

        # since there is only one measurement in each image, both assigned feature index 0
        matched_idxs = np.array([[0,0]])

        # create matches
        matches = {img_idxs: matched_idxs}
        da = DataAssociation(matches, feature_list)
        triangulated_landmark_map = da.run(self.poses, False, 5, 2, False, None, cameras)

        computed_landmark = triangulated_landmark_map[0].point3()
        self.gtsamAssertEquals(computed_landmark, self.expected_landmark,1e-2)

    
    def test_create_computation_graph(self):
        """
        Tests the graph to create data association for images. 
        """
        sharedCal = gtsam.Cal3Bundler(1500, 0, 0, 640, 480)
        matches, features, poses = self.__generate_3_poses(sharedCal)
        # Run without computation graph
        da = DataAssociation(matches, features)
        expected_landmark_map = da.run(poses, True, 5, 3, False, sharedCal, None)

        # Run with computation graph
        computed_landmark_map = da.create_computation_graph(poses, True, 5, 3, False, sharedCal, None)

        with dask.config.set(scheduler='single-threaded'):
            dask_result = dask.compute(computed_landmark_map)[0]

        assert len(expected_landmark_map) == len(dask_result), "Dask not configured correctly"

        for i in range(len(expected_landmark_map)):
            assert expected_landmark_map[i].number_measurements() == dask_result[i].number_measurements(), "Dask tracks incorrect"
            # Test if the measurement in both are equal
            np.testing.assert_array_almost_equal(expected_landmark_map[i].measurement(0)[1], dask_result[i].measurement(0)[1], 1, "Dask measurements incorrect")
        
        
    def __generate_2_poses(self, sharedCal):
        """
        Generate 2 matches and their corresponding poses for shared calibration
        """
        # Amount of noise to be added to measurements
        noise_params = (-np.array([0.1, 0.5]), - np.array([-0.2, 0.3]))

        measurements, feature_list, img_idxs, _ = self.__generate_measurements((sharedCal, sharedCal), noise_params, self.poses)     

        # since there is only one measurement in each image, both assigned feature index 0
        matched_idxs = np.array([[0,0]])

        # create matches
        matches_1 = {img_idxs: matched_idxs}
        return matches_1, feature_list, self.poses, measurements

    def __generate_3_poses(self, sharedCal):
        """
        Generate 3 matches and corresponding poses with shared calibration
        """
        matches, feature_list, poses, measurements = self.__generate_2_poses(sharedCal)

        # Add third camera slightly rotated
        rotatedCamera = gtsam.Rot3.Ypr(0.1, 0.2, 0.1)
        pose3 = gtsam.Pose3(rotatedCamera, gtsam.Point3(0.1, -2, -0.1))
        camera3 = gtsam.PinholeCameraCal3Bundler(pose3, sharedCal)
        z3 = camera3.project(self.expected_landmark)
        # add noise to measurement
        measurements.append(z3 + np.array([0.1, -0.1]))
        poses.append(pose3)

        img_idxs2 = tuple(list(range(1, len(poses))))
        obs_in_img = []
        obs_in_img.append(tuple(measurements[2]))
        feature_list.append(obs_in_img)
        
        # Only one measurement in images 1 and 2, hence each get index 0
        matched_idxs2 = np.array([[0,0]])
        match_dict = {img_idxs2: matched_idxs2}
        matches.update(match_dict)
        return matches, feature_list, poses

    def __generate_measurements(self, calibration, noise_params, poses):
        """ Generate measurements for given calibration and poses """
        measurements = gtsam.Point2Vector()
        cameras = gtsam.CameraSetCal3Bundler()
        for i in range(len(poses)):
            camera = gtsam.PinholeCameraCal3Bundler(poses[i], calibration[i])
            # Project landmark into two cameras and triangulate
            z = camera.project(self.expected_landmark)
            cameras.append(camera)
            measurements.append(z + noise_params[i])
        # Create image indices for each pose
        img_idxs = tuple(list(range(len(self.poses))))
        # List of features in each image
        feature_list = []
        for i in range(len(measurements)):
            obs_in_img = []
            obs_in_img.append(tuple(measurements[i]))
            feature_list.append(obs_in_img)
        return measurements, feature_list, img_idxs, cameras   
        
if __name__ == "__main__":
    unittest.main()