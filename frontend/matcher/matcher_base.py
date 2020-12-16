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

    def create_computation_graph(self,
                                 image_pair_indices: Tuple[int, int],
                                 descriptor_graph: List[Delayed],
                                 distance_type: MatchingDistanceType = MatchingDistanceType.EUCLIDEAN
                                 ) -> Delayed:
        """
        Generates computation graph for matched features using the detection and description graph.

        Args:
            image_pair_indices: valid pair of images which are to be matched.
            description: list of dask tasks for descriptor generation for each image.
            distance_type (optional): the space to compute the distance between
                                      descriptors. Defaults to
                                      MatchingDistanceType.EUCLIDEAN.

        Returns:
            Delayed dask tasks for matching for input camera pairs.
        """
        i1, i2 = image_pair_indices
        delayed_matches_graph = dask.delayed(self.match)(
                descriptor_graph[i1],
                descriptor_graph[i2],
                distance_type
            )

        return delayed_matches_graph
