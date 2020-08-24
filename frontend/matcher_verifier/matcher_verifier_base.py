"""
Base class for the M+V (matching+verification) stage of the frontend.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Tuple

import dask
import numpy as np


class MatcherVerifierBase(metaclass=abc.ABCMeta):
    """
    Base class for all methods which provide a joint matching plus verification API.

    The API taking features and their descriptors from two images as input and returns the computed fundamental matrix and the list of valid correspondences.
    """

    @abc.abstractmethod
    def match_and_verify(self,
                         features_im1: np.ndarray,
                         features_im2: np.ndarray,
                         descriptors_im1: np.ndarray,
                         descriptors_im2: np.ndarray,
                         image_shape_im1: Tuple[int, int],
                         image_shape_im2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Matches the features (using their corresponding descriptors) to return geometrically verified outlier-free correspondences as indices of input features.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corresponding descriptors from image #1
            descriptors_im2 (np.ndarray): corresponding descriptors from image #1
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2

        Returns:
            np.ndarray: estimated fundamental matrix
            np.ndarray: verified correspondences as index of the input features in a Nx2 array
        """

    def match_and_verify_and_get_features(self,
                                          features_im1: np.ndarray,
                                          features_im2: np.ndarray,
                                          descriptors_im1: np.ndarray,
                                          descriptors_im2: np.ndarray,
                                          image_shape_im1: Tuple[int, int],
                                          image_shape_im2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs the match_and_verify functionality and fetches the actual features corresponding to indices of verified correspondenses.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corresponding descriptors from image #1
            descriptors_im2 (np.ndarray): corresponding descriptors from image #1
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2

        Returns:
            np.ndarray: estimated fundamental matrix
            np.ndarray: the features from image #1 which serve as verified correspondences.
            np.ndarray: the corresponding features from image #2 which serve as verified correspondences.
        """
        F, correspondence_indices = self.match_and_verify(
            features_im1, features_im2,
            descriptors_im1, descriptors_im2,
            image_shape_im1, image_shape_im2)

        return F, features_im1[correspondence_indices[:, 0]], features_im2[correspondence_indices[:, 1]]

    def create_computation_graph(self,
                                 detection_description_graph: List[dask.delayed],
                                 image_shapes: List[Tuple[int, int]]
                                 ) -> Dict[Tuple[int, int], dask.delayed]:
        """
        Created the computation graph for verification using the graph from matcher stage

        Args:
            matcher_graph (Dict[Tuple[int, int], dask.delayed]): computation graph from matcher
            image_shapes (List[Tuple[int, int]]): list of all image shapes

        Returns:
            Dict[Tuple[int, int], dask.delayed]: delayed dask tasks for verification
        """

        result = dict()

        num_images = len(detection_description_graph)

        for idx1 in range(num_images):
            for idx2 in range(idx1+1, num_images):
                graph_component_im1 = detection_description_graph[idx1]
                graph_component_im2 = detection_description_graph[idx2]

                result[(idx1, idx2)] = dask.delayed(self.match_and_verify_and_get_features)(
                    graph_component_im1[0], graph_component_im2[0],
                    graph_component_im1[1], graph_component_im2[1],
                    image_shapes[idx1],
                    image_shapes[idx2]
                )

        return result
