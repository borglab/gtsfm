"""Unit test for the DataAssociation class.

Authors:Sushmita Warrier
"""
from random import uniform
import unittest

from collections import defaultdict
import dask
import numpy as np
import gtsam
from gtsam.utils.test_case import GtsamTestCase

import utils.io as io_utils
from data_association.data_assoc import DataAssociation, LandmarkInitialization
from data_association.tracks import FeatureTracks, toy_case, delete_malformed_tracks
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

        # self.da = DataAssociation()
        self.matcher = DummyMatcher()
    
    def test_track(self):
        """
        Tests that the tracks are being merged and mapped correctly
        """
        dummy_matches = toy_case()
        self.track = FeatureTracks(dummy_matches, len(dummy_matches), None)
        # len(track) value for toy case strictly
        assert len(self.track.landmark_map) == 4, "tracks incorrectly mapped"

    
    def test_primary_filtering(self):
        """
        Tests that the tracks are being filtered correctly.
        Removes tracks that have two measurements in a single image.
        """
        malformed_landmark_map = {
            (0, 0): [(0, (1, 3)), (1, (12, 14)), (1, (8, 2)), (2, (13, 16))], 
            (0, 1): [(0, (4, 6)), (1, (5, 10)), (2, (12, 14))], 
            (0, 2): [(0, (9, 8)), (0,(2,4)), (1, (11, 12))], 
            (1, 7): [(1, (4, 1)), (2, (8, 1))]}
        expected_landmark_map = {
            (0, 1): [(0, (4, 6)), (1, (5, 10)), (2, (12, 14))], 
            (1, 7): [(1, (4, 1)), (2, (8, 1))]
        }
        filtered_map = delete_malformed_tracks(malformed_landmark_map)
        for key, _ in filtered_map.items():
            # check that the length of the observation list corresponding to each key is the same. Only good tracks will remain
            assert len(expected_landmark_map[key]) == len(filtered_map[key]), "Tracks not filtered correctly"

    def test_triangulation_sharedCal(self):
        """
        Tests that the triangulation is accurate. 
        Example from borglab/gtsam/python/gtsam/tests/test_Triangulation.py
        and gtsam/geometry/tests/testTriangulation.cpp
        """  
        sharedCal = gtsam.Cal3_S2(1500, 1200, 0, 640, 480)

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

        # assuming same nb of images as nb of poses
        img_idxs = tuple(list(range(len(poses))))
        match_arrays = tuple((np.expand_dims(np.asarray(measurements[0]), axis=0), np.expand_dims(np.asarray(measurements[1]), axis=0)))

        # create matches
        matches_1 = {img_idxs: match_arrays}

        computed_landmark = DataAssociation(matches_1, len(poses), poses, True, sharedCal, None).triangulated_landmarks[0]
        self.gtsamAssertEquals(computed_landmark, expected_landmark,1e-1)

        # Add third camera slightly rotated
        rotatedCamera = gtsam.Rot3.Ypr(0.1, 0.2, 0.1)
        pose3 = gtsam.Pose3(rotatedCamera, gtsam.Point3(0.1, -2, -0.1))
        camera3 = gtsam.PinholeCameraCal3_S2(pose3, sharedCal)
        z3 = camera3.project(expected_landmark)
        # add noise to measurement
        measurements.append(z3 + np.array([0.1, -0.1]))
        poses.append(pose3)

        img_idxs2 = tuple(list(range(1, len(poses))))
        match_arrays2 = tuple(( 
            np.expand_dims(np.asarray(measurements[1]), axis=0), 
            np.expand_dims(np.asarray(measurements[2]), axis=0)))

        matches_2 = {img_idxs: match_arrays, img_idxs2: match_arrays2}
        computed_landmark = DataAssociation(matches_2, len(poses), poses, True, sharedCal, None).triangulated_landmarks[0]
        self.gtsamAssertEquals(computed_landmark, expected_landmark,1e-1)

    
    def test_create_computation_graph(self):
        """
        Test currently incomplete.
        Tests the graph to create data association for images. 
        """

        correspondences = self.get_random_correspondences()  # gives matches for 3 images only
        pose_list = self.generate_poses(len(correspondences))
        self.track_list = FeatureTracks(correspondences, len(correspondences), pose_list)
        

    def __generate_rand_binary_descs(self, num_descriptors: int, descriptor_length: int) -> np.ndarray:
        """
        Generates random binary descriptors.

        Args:
            num_descriptors: number of descriptors to generate
            descriptor_length: length of each descriptor vector

        Returns:
            generated descriptor
        """
        if num_descriptors == 0:
            return np.array([], dtype=np.uint8)

        return np.random.randint(
            0, high=2, size=(num_descriptors, descriptor_length)
        ).astype(np.uint8)

    def get_random_correspondences(self):
        """
        Gets random verified correspondences
        """
        # generate three random descriptors and their features
        num_descriptors_im1 = np.random.randint(5, 15)
        num_descriptors_im2 = np.random.randint(5, 15)
        num_descriptors_im3 = np.random.randint(5, 15)
        print("num descr", num_descriptors_im1, num_descriptors_im2, num_descriptors_im3)

        descriptor_length = np.random.randint(2, 10)
        # print("descr length", descriptor_length)

        descriptor_list = [
            self.__generate_rand_binary_descs(
                num_descriptors_im1, descriptor_length),
            self.__generate_rand_binary_descs(
                num_descriptors_im2, descriptor_length),
            self.__generate_rand_binary_descs(
                num_descriptors_im3, descriptor_length)
        ]


        features_list = [
            np.random.rand(num_descriptors_im1, 3),
            np.random.rand(num_descriptors_im2, 3),
            np.random.rand(num_descriptors_im3, 3),
        ]

        # create computation graphs
        detection_description_graph = [
            dask.delayed((x, y)) for x, y in zip(features_list, descriptor_list)
        ]

        matcher_graph = self.matcher.create_computation_graph(
            detection_description_graph)
        
        # run it in sequential mode
        results = []
        with dask.config.set(scheduler='single-threaded'):
            results = dask.compute(matcher_graph)[0]

        return results
    
    def generate_poses(self, nb_poses:int) -> List:
        """
        Generate random poses
        Args:
            nb_poses: nb of poses to be created
        Returns:
            pose_list: list of poses
        """
        list_absolute_t = [gtsam.Point3(
        np.random.uniform(100.2, 120.5), 
        np.random.uniform(150.9, 140.5), 
        np.random.uniform(12.2, 20.5)) for _ in range(nb_poses)]

        list_absolute_R = [gtsam.Rot3(1.0, 0.0, 0.0, 
                                      0.0, 1.0, 0.0, 
                                      0.0, 0.0, 1.0)] # first one is identity
        for _ in range(1, nb_poses):
            list_absolute_R.append(gtsam.Rot3(
                np.random.rand(), np.random.rand(), np.random.rand(), 
                np.random.rand(), np.random.rand(), np.random.rand(),
                np.random.rand(), np.random.rand(), np.random.rand()))

        pose_list = [gtsam.Pose3(R, t) for R, t in zip(list_absolute_R, list_absolute_t)]
        return pose_list
    
        
        
if __name__ == "__main__":
    unittest.main()