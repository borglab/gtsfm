"""Base class for the M (matcher) stage of the front end.

Authors: Ayush Baid
"""
import abc
from enum import Enum
from typing import Dict, List, Tuple

import dask
import numpy as np
from dask.delayed import Delayed


class MatchingDistanceType(Enum):
    """Type of distance metric to use for matching descriptors."""
    HAMMING = 1
    EUCLIDEAN = 2


class MatcherBase(metaclass=abc.ABCMeta):
    """Base class for all matchers.

    Matchers work on a pair of descriptors and match them by their distance.
    """

    @abc.abstractmethod
    def match(self,
              descriptors_im1: np.ndarray,
              descriptors_im2: np.ndarray,
              distance_type: MatchingDistanceType =
              MatchingDistanceType.EUCLIDEAN) -> np.ndarray:
        """Match descriptor vectors.

        Output format:
        1. Each row represents a match.
        2. First column represents descriptor index from image #1.
        3. Second column represents descriptor index from image #2.
        4. Matches are sorted in descending order of the confidence (score).

        Args:
            descriptors_im1: descriptors from image #1, of shape (N1, D).
            descriptors_im2: descriptors from image #2, of shape (N2, D).
            distance_type (optional): the space to compute the distance between
                                      descriptors. Defaults to
                                      MatchingDistanceType.EUCLIDEAN.

        Returns:
            Match indices (sorted by confidence), as matrix of shape
                (N, 2), where N < min(N1, N2).
        """
        # TODO(ayush): should I define matcher on descriptors or the distance matrices.
        # TODO(ayush): how to handle deep-matchers which might require the full image as input

    def match_and_get_features(self,
                               features_im1: np.ndarray,
                               features_im2: np.ndarray,
                               descriptors_im1: np.ndarray,
                               descriptors_im2: np.ndarray,
                               distance_type: MatchingDistanceType =
                               MatchingDistanceType.EUCLIDEAN
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Match descriptors and return the corresponding features.

        Args:
            features_im1: features from image #1, of shape (N1, 2+).
            features_im2: features from image #2, of shape (N2, 2+).
            descriptors_im1: descriptors for features_im1, of shape (N1, D).
            descriptors_im2: descriptors for features_im1, of shape (N2, D).
            distance_type (optional): the space to compute the distance between
                                      descriptors. Defaults to
                                      MatchingDistanceType.EUCLIDEAN.

        Returns:
            Features from im1 and their corresponding matched features from im2,
                each having shape (N, 2), where N < min(N1, N2).
        """
        match_indices = self.match(
            descriptors_im1, descriptors_im2, distance_type)

        return features_im1[match_indices[:, 0], :], \
            features_im2[match_indices[:, 1], :]

    def create_computation_graph(self,
                                 pair_indices: List[Tuple[int, int]],
                                 detection_description_graph: List[Delayed],
                                 distance_type: MatchingDistanceType =
                                 MatchingDistanceType.EUCLIDEAN
                                 ) -> Dict[Tuple[int, int], Delayed]:
        """
        Generates computation graph for matched features using the detection and description graph.

        Args:
            pair_indices: valid pairs in input scene which are to be matched.
            detection_description_graph: computation graph for features and
                                         their associated descriptors for each
                                         image.
            distance_type (optional): the space to compute the distance between
                                      descriptors. Defaults to
                                      MatchingDistanceType.EUCLIDEAN.

        Returns:
            Delayed dask tasks for matching for input camera pairs.
        """

        graph = dict()

        for idx1, idx2 in pair_indices:

            graph_component_im1 = detection_description_graph[idx1]
            graph_component_im2 = detection_description_graph[idx2]

            graph[(idx1, idx2)] = dask.delayed(self.match)(
                graph_component_im1[1], graph_component_im2[1],
                distance_type
            )

        return graph
