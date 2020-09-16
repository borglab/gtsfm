"""Unit test for the DataAssociation class.

Authors:Sushmita Warrier
"""
from random import uniform
import unittest

import dask
import numpy as np
import gtsam

import utils.io as io_utils
from data_association.data_assoc import DataAssociation, LandmarkInitialization
from data_association.tracks import FeatureTracks, toy_case
from frontend.matcher.dummy_matcher import DummyMatcher


class TestDataAssociation(unittest.TestCase):
    """
    Unit tests for data association, which maps the feature tracks to their 3D landmarks.
    """

    def setUp(self):
        """
        Set up the data association module.
        """
        super().setUp()

        # set up ground truth data for comparison

        self.da = DataAssociation()
        self.matcher = DummyMatcher()
    
    def test_track(self):
        """
        Tests that the tracks are being merged and mapped correctly
        """
        dummy_matches = toy_case()
        self.track = FeatureTracks(dummy_matches, len(dummy_matches), None)
        # len(track) value for toy case strictly
        assert len(self.track.landmark_map) == 4, "tracks incorrectly mapped"

    def test_filtering(self):
        """
        Tests that the tracks are being filtered correctly
        """
        pass


    def test_create_computation_graph(self):
        """
        Tests the graph to create data association for images
        """

        correspondences = self.get_random_correspondences()  # gives matches for 3 images only
        pose_list = self.generate_poses(len(correspondences))
        self.track_list = FeatureTracks(correspondences, len(correspondences), pose_list)
        

    def __generate_random_binary_descriptors(self, num_descriptors: int, descriptor_length: int) -> np.ndarray:
        """
        Generates random binary descriptors.

        Args:
            num_descriptors (int): number of descriptors to generate
            descriptor_length (int): length of each descriptor vector

        Returns:
            np.ndarray: generated descriptor
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
            self.__generate_random_binary_descriptors(
                num_descriptors_im1, descriptor_length),
            self.__generate_random_binary_descriptors(
                num_descriptors_im2, descriptor_length),
            self.__generate_random_binary_descriptors(
                num_descriptors_im3, descriptor_length)
        ]

        # print("descr list", descriptor_list[0])

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
    
    def generate_poses(self, len_poses):
        list_absolute_t = [gtsam.Point3(
        np.random.uniform(100.2, 120.5), 
        np.random.uniform(150.9, 140.5), 
        np.random.uniform(12.2, 20.5)) for _ in range(len_poses)]

        list_absolute_R = [gtsam.Rot3(1.0, 0.0, 0.0, 
                                      0.0, 1.0, 0.0, 
                                      0.0, 0.0, 1.0)] # first one is identity
        for _ in range(1, len_poses):
            list_absolute_R.append(gtsam.Rot3(
                np.random.rand(), np.random.rand(), np.random.rand(), 
                np.random.rand(), np.random.rand(), np.random.rand(),
                np.random.rand(), np.random.rand(), np.random.rand()))

        pose_list = [gtsam.Pose3(R, t) for R, t in zip(list_absolute_R, list_absolute_t)]
        return pose_list
    
    def test_initial_estimates(self):
        """
        Tests initial estimates
        """
        dummy_matches = toy_case()
        sharedCal = gtsam.Cal3_S2(1500, 1200, 0, 640, 480) 
        track = FeatureTracks(dummy_matches, len(dummy_matches), None)
        pose_list = self.generate_poses(3)
        LI = LandmarkInitialization(sharedCal, pose_list)
        
        
if __name__ == "__main__":
    unittest.main()