"""Unit test for the DataAssociation class.

Authors: Sushmita Warrier
"""
from random import uniform
import unittest

import dask
import numpy as np
import gtsam
from gtsam.utils.test_case import GtsamTestCase
from typing import List, Dict

from data_association.data_assoc import DataAssociation, LandmarkInitialization
from data_association.feature_tracks import FeatureTrackGenerator
from frontend.matcher.dummy_matcher import DummyMatcher


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
        self.matcher = DummyMatcher()
    
    def test_track(self):
        """
        Tests that the tracks are being merged and mapped correctly
        """
        self.track = FeatureTrackGenerator(self.dummy_matches, len(self.dummy_matches), self.feature_list)
        # len(track) value for toy case strictly
        assert len(self.track.filtered_landmark_data) == 4, "tracks incorrectly mapped"

    
    def test_primary_filtering(self):
        """
        Tests that the tracks are being filtered correctly.
        Removes tracks that have two measurements in a single image.
        """

        filtered_map = FeatureTrackGenerator(self.malformed_matches, len(self.malformed_matches), self.feature_list).filtered_landmark_data

        # check that the length of the observation list corresponding to each key is the same. Only good tracks will remain
        assert len(filtered_map) == 4, "Tracks not filtered correctly"

    def test_triangulation_sharedCal(self):
        """
        Tests that the triangulation is accurate. 
        Example from borglab/gtsam/python/gtsam/tests/test_Triangulation.py
        and gtsam/geometry/tests/testTriangulation.cpp
        """  
        sharedCal = gtsam.Cal3_S2(1500, 1200, 0, 640, 480)

        matches_1, feature_list, poses, _ = self.__generate_2_poses(sharedCal)
        da = DataAssociation()
        triangulated_landmark_map = da.run(matches_1, len(poses), poses, True, sharedCal, None, feature_list)
        assert len(triangulated_landmark_map) == 0, "tracks exceeding expected track length"
        

        matches_2, feature_list, poses = self.__generate_3_poses(sharedCal)
        da = DataAssociation()
        triangulated_landmark_map = da.run(matches_2, len(poses), poses, True, sharedCal, None, feature_list)
        computed_landmark = triangulated_landmark_map[0].point3()
        # landmark ~5 meters infront of camera, computed from __generate_X_poses
        expected_landmark = gtsam.Point3(5, 0.5, 1.2)
        assert len(triangulated_landmark_map)== 1, "more tracks than expected"
        self.gtsamAssertEquals(computed_landmark, expected_landmark,1e-1)

    
    def test_create_computation_graph(self):
        """
        Tests the graph to create data association for images. 
        """
        sharedCal = gtsam.Cal3_S2(1500, 1200, 0, 640, 480)
        matches, features, poses = self.__generate_3_poses(sharedCal)
        # Run without computation graph
        da = DataAssociation()
        expected_landmark_map = da.run(matches, len(poses), poses, True, sharedCal, None, features)

        # Run with computation graph
        da = DataAssociation()
        computed_landmark_map = da.create_computation_graph(matches, len(poses), poses, True, sharedCal, None, features)


        with dask.config.set(scheduler='single-threaded'):
            dask_result = dask.compute(computed_landmark_map)[0]

        assert len(expected_landmark_map) == len(dask_result), "Dask not configured correctly"
        for i in range(len(expected_landmark_map)):
            assert expected_landmark_map[i].number_measurements() == dask_result[i].number_measurements(), "Dask tracks incorrect"
            # Test if the measurement in both are equal
            np.testing.assert_array_almost_equal(expected_landmark_map[i].measurement(0)[1], dask_result[i].measurement(0)[1], 1, "Dask measurements incorrect")
        
        
    def __generate_2_poses(self, sharedCal):
        """
        Generate two matches and their corresponding poses
        """
        # Looking along X-axis, 1 meter above ground plane (x-y)
        upright = gtsam.Rot3.Ypr(-np.pi / 2, 0., -np.pi / 2)
        pose1 = gtsam.Pose3(upright, gtsam.Point3(0, 0, 1))
        camera1 = gtsam.PinholeCameraCal3_S2(pose1, sharedCal)

        # create second camera 1 meter to the right of first camera
        pose2 = pose1.compose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0)))
        camera2 = gtsam.PinholeCameraCal3_S2(pose2, sharedCal)

        # landmark ~5 meters infront of camera
        expected_landmark = gtsam.Point3(5, 0.5, 1.2)

        # Project landmark into two cameras and triangulate
        z1 = camera1.project(expected_landmark)
        z2 = camera2.project(expected_landmark)

        poses = gtsam.Pose3Vector()
        measurements = gtsam.Point2Vector()
        poses.append(pose1)
        poses.append(pose2)
        # Add some noise - computed landmark should be ~ (4.995, 0.499167, 1.19814)        
        measurements.append(z1 - np.array([0.1, 0.5]))
        measurements.append(z2 - np.array([-0.2, 0.3]))
        # since there is only one measurement in each image, both assigned feature index 0
        matched_idxs = np.array([[0,0]])

        # assuming same nb of images as nb of poses
        img_idxs = tuple(list(range(len(poses))))
        feature_list = []
        for i in range(len(measurements)):
            msrmnt_in_img = []
            msrmnt_in_img.append(tuple(measurements[i]))
            feature_list.append(msrmnt_in_img)

        # create matches

        matches_1 = {img_idxs: matched_idxs}
        return matches_1, feature_list, poses, measurements

    def __generate_3_poses(self, sharedCal):
        matches, feature_list, poses, measurements = self.__generate_2_poses(sharedCal)
        expected_landmark = gtsam.Point3(5, 0.5, 1.2)

        # Add third camera slightly rotated
        rotatedCamera = gtsam.Rot3.Ypr(0.1, 0.2, 0.1)
        pose3 = gtsam.Pose3(rotatedCamera, gtsam.Point3(0.1, -2, -0.1))
        camera3 = gtsam.PinholeCameraCal3_S2(pose3, sharedCal)
        z3 = camera3.project(expected_landmark)
        # add noise to measurement
        measurements.append(z3 + np.array([0.1, -0.1]))
        poses.append(pose3)

        img_idxs2 = tuple(list(range(1, len(poses))))
        msrmnt_in_img = []
        msrmnt_in_img.append(tuple(measurements[2]))
        feature_list.append(msrmnt_in_img)
        
        # Only one measurement in images 1 and 2, hence each get index 0
        matched_idxs2 = np.array([[0,0]])
        match_dict = {img_idxs2: matched_idxs2}
        matches.update(match_dict)
        return matches, feature_list, poses

    # def __generate_rand_binary_descs(self, num_descriptors: int, descriptor_length: int) -> np.ndarray:
    #     """
    #     Generates random binary descriptors.

    #     Args:
    #         num_descriptors: number of descriptors to generate
    #         descriptor_length: length of each descriptor vector

    #     Returns:
    #         generated descriptor
    #     """
    #     if num_descriptors == 0:
    #         return np.array([], dtype=np.uint8)

    #     return np.random.randint(
    #         0, high=2, size=(num_descriptors, descriptor_length)
    #     ).astype(np.uint8)

    # def get_random_correspondences(self):
    #     """
    #     Gets random verified correspondences
    #     """
    #     # generate three random descriptors and their features
    #     num_descriptors_im1 = np.random.randint(5, 15)
    #     num_descriptors_im2 = np.random.randint(5, 15)
    #     num_descriptors_im3 = np.random.randint(5, 15)
    #     print("num descr", num_descriptors_im1, num_descriptors_im2, num_descriptors_im3)

    #     descriptor_length = np.random.randint(2, 10)
    #     # print("descr length", descriptor_length)

    #     descriptor_list = [
    #         self.__generate_rand_binary_descs(
    #             num_descriptors_im1, descriptor_length),
    #         self.__generate_rand_binary_descs(
    #             num_descriptors_im2, descriptor_length),
    #         self.__generate_rand_binary_descs(
    #             num_descriptors_im3, descriptor_length)
    #     ]


    #     features_list = [
    #         np.random.rand(num_descriptors_im1, 3),
    #         np.random.rand(num_descriptors_im2, 3),
    #         np.random.rand(num_descriptors_im3, 3),
    #     ]

    #     # create computation graphs
    #     detection_description_graph = [
    #         dask.delayed((x, y)) for x, y in zip(features_list, descriptor_list)
    #     ]

    #     matcher_graph = self.matcher.create_computation_graph(
    #         detection_description_graph)
        
    #     # run it in sequential mode
    #     results = []
    #     with dask.config.set(scheduler='single-threaded'):
    #         results = dask.compute(matcher_graph)[0]

    #     return results
    
    # def generate_poses(self, nb_poses:int) -> List:
    #     """
    #     Generate random poses
    #     Args:
    #         nb_poses: nb of poses to be created
    #     Returns:
    #         pose_list: list of poses
    #     """
    #     list_absolute_t = [gtsam.Point3(
    #     np.random.uniform(100.2, 120.5), 
    #     np.random.uniform(150.9, 140.5), 
    #     np.random.uniform(12.2, 20.5)) for _ in range(nb_poses)]

    #     list_absolute_R = [gtsam.Rot3(1.0, 0.0, 0.0, 
    #                                   0.0, 1.0, 0.0, 
    #                                   0.0, 0.0, 1.0)] # first one is identity
    #     for _ in range(1, nb_poses):
    #         list_absolute_R.append(gtsam.Rot3(
    #             np.random.rand(), np.random.rand(), np.random.rand(), 
    #             np.random.rand(), np.random.rand(), np.random.rand(),
    #             np.random.rand(), np.random.rand(), np.random.rand()))

    #     pose_list = [gtsam.Pose3(R, t) for R, t in zip(list_absolute_R, list_absolute_t)]
    #     return pose_list
    
        
        
if __name__ == "__main__":
    unittest.main()