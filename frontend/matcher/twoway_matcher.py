"""
Two way (mutual nearest neighbor) matcher.

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

from frontend.matcher.matcher_base import MatcherBase


class TwoWayMatcher(MatcherBase):
    """
    Two way (mutual nearest neighbor) matcher using OpenCV.
    """

    def __init__(self, distance_type='euclidean'):
        super().__init__()

        if distance_type == 'euclidean':
            self.distance_metric = cv.NORM_L2
        elif distance_type == 'hamming':
            self.distance_metric = cv.NORM_HAMMING
        else:
            raise NotImplementedError(
                'The specified distance type is not implemented')

    def match(self,
              descriptors_im1: np.ndarray,
              descriptors_im2: np.ndarray) -> np.ndarray:
        """
        Match a pair of descriptors.

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
        valid_idx_im1 = np.nonzero(~(np.isnan(descriptors_im1).any(axis=1)))[0]
        valid_idx_im2 = np.nonzero(~(np.isnan(descriptors_im2).any(axis=1)))[0]

        descriptors_1 = descriptors_im1[valid_idx_im1]
        descriptors_2 = descriptors_im2[valid_idx_im2]

        # run OpenCV's matcher
        bf = cv.BFMatcher(normType=self.distance_metric, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda r: r.distance)

        match_indices = np.array(
            [[m.queryIdx, m.trainIdx] for m in matches]).astype(np.int32)

        if match_indices.size == 0:
            return np.array([])

        # remap them back
        match_indices[:, 0] = valid_idx_im1[match_indices[:, 0]]
        match_indices[:, 1] = valid_idx_im2[match_indices[:, 1]]

        return match_indices
