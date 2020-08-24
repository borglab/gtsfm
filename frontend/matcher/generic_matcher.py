"""
A generic matcher class with three basics type of matching:
1. Two way matching
2. One way matching w/ ratio test: greedy 1:1
3. One way matching w/o ratio test: greedy 1:1

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np
import sklearn.metrics

import utils.generic_utils as generic_utils
from frontend.matcher.matcher_base import MatcherBase


class GenericMatcher(MatcherBase):
    """
    Generic Matcher using OpenCV for matching
    """

    def __init__(self, ratio_threshold=1.1, is_two_way=True, distance_type='euclidean'):
        super().__init__()

        self.ratio_threshold = ratio_threshold

        # we perform ratio test only when the ratio is a valid number
        self.perform_ratio_test = self.ratio_threshold < 1.0

        self.is_two_way = is_two_way

        self.distance_type = distance_type

        if self.distance_type == 'euclidean':
            self.distance_metric = cv.NORM_L2
        elif self.distance_type == 'hamming':
            self.distance_metric = cv.NORM_HAMMING
        else:
            raise NotImplementedError(
                'The specified distance type is not implemented')

    def match(self,
              descriptors_im1: np.ndarray,
              descriptors_im2: np.ndarray) -> np.ndarray:
        """
        Match descriptors from two images.

        Refer to documentation in the parent class for detailed output format.

        Args:
            descriptors_im1 (np.ndarray): descriptors from image #1
            descriptors_im2 (np.ndarray): descriptors from image #2

        Returns:
            np.ndarray: match indices (sorted by confidence)
        """

        if descriptors_im1.size == 0 or descriptors_im2.size == 0:
            return np.array([])

         # we will have to remove NaNs by ourselves
        idx_map_1 = np.nonzero(~(np.isnan(descriptors_im1).any(axis=1)))[0]
        idx_map_2 = np.nonzero(~(np.isnan(descriptors_im2).any(axis=1)))[0]

        descriptors_1 = descriptors_im1[idx_map_1]
        descriptors_2 = descriptors_im2[idx_map_2]

        if self.is_two_way or self.perform_ratio_test:
            bf = cv.BFMatcher(normType=self.distance_metric,
                              crossCheck=self.is_two_way)

            if self.distance_metric == cv.NORM_HAMMING:
                descriptors_1 = descriptors_1.astype(np.uint8)
                descriptors_2 = descriptors_2.astype(np.uint8)

            if self.perform_ratio_test:
                matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

                ratio_test_results = [
                    m[0].distance <= m[1].distance*self.ratio_threshold if len(m) > 0 else False for m in matches
                ]

                match_indices = self.__greedy_nn_match(
                    descriptors_1[ratio_test_results], descriptors_2)

                if match_indices.size == 0:
                    return np.array([])

                # map the new indices back to the original indices
                valid_descriptors_idxes = np.nonzero(ratio_test_results)[0]

                match_indices[:, 0] = \
                    valid_descriptors_idxes[match_indices[:, 0]]

            else:

                match_indices = []

                matches = bf.match(descriptors_1, descriptors_2)

                matches = sorted(matches, key=lambda r: r.distance)

                match_indices = np.array(
                    [[m.queryIdx, m.trainIdx] for m in matches]).astype(np.int32)

        else:

            match_indices = self.__greedy_nn_match(
                descriptors_1, descriptors_2)

        if match_indices.size == 0:
            return np.array([])

        # remap them back
        match_indices[:, 0] = idx_map_1[match_indices[:, 0]]
        match_indices[:, 1] = idx_map_2[match_indices[:, 1]]

        return match_indices

    def __greedy_nn_match(self, descriptors_im1, descriptors_im2):
        if descriptors_im1.size == 0 or descriptors_im2.size == 0:
            return np.array([])

        dist_matrix = sklearn.metrics.pairwise_distances(
            descriptors_im1, descriptors_im2, metric=self.distance_type)

        match_indices, _ = generic_utils.find_closest_match_greedy(
            dist_matrix, dist_threshold=None)

        return match_indices
