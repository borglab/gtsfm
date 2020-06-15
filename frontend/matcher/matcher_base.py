"""
Base class for the M (matcher) stage of the front end.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Tuple

import dask
import numpy as np


class MatcherBase(metaclass=abc.ABCMeta):
    """
    Base class for all matchers.
    Matchers work on a pair of descriptors and match them by their distance.
    """

    @abc.abstractmethod
    def match(self, descriptors_im1: np.ndarray, descriptors_im2: np.ndarray) -> np.ndarray:
        """
        Match a pair of descriptors.

        Output format:
        1. Each row represents a match
        2. The entry in first column represents the index of a descriptor from image #1
        3. The entry in first column represents the index of a descriptor from image #2
        4. The matches are sorted in descending order of the confidence (score)

        Args:
            descriptors_im1 (np.ndarray): descriptors from image #1
            descriptors_im2 (np.ndarray): descriptors from image #2

        Returns:
            np.ndarray: match indices (sorted by confidence)
        """
        # TODO(ayush): should I define matcher on descriptors or the distance matrices.
        # TODO(ayush): how to handle deep-matchers which might require the full image as input

    def match_and_get_features(self,
                               features_im1: np.ndarray,
                               features_im2: np.ndarray,
                               descriptors_im1: np.ndarray,
                               descriptors_im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match descriptors and return the corresponding features.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corresponding descriptors from image #1
            descriptors_im2 (np.ndarray): corresponding descriptors from image #2

        Returns:
            Tuple[np.ndarray, np.ndarray]: matched features from each image
        """
        match_indices = self.match(descriptors_im1, descriptors_im2)

        return features_im1[match_indices[:, 0], :], features_im2[match_indices[:, 1], :]

    def create_computation_graph(self,
                                 detection_description_graph: List[dask.delayed],
                                 ) -> Dict[Tuple[int, int], dask.delayed]:
        """
        Generates computation graph for matched features using the detection and description graph.

        Args:
            detection_description_graph (List[dask.delayed]): computation graph for features and
                                                              their associated descriptors for each image

        Returns:
            Dict[Tuple[int, int], dask.delayed]: delayed dask tasks for match_and_get_features
                                                 for each tuple of input indices
        """

        result = dict()

        num_images = len(detection_description_graph)

        for idx1 in range(num_images):
            for idx2 in range(idx1+1, num_images):
                graph_component_im1 = detection_description_graph[idx1]
                graph_component_im2 = detection_description_graph[idx2]

                result[(idx1, idx2)] = dask.delayed(self.match_and_get_features)(
                    graph_component_im1[0], graph_component_im2[0],
                    graph_component_im1[1], graph_component_im2[1]
                )

        return result
