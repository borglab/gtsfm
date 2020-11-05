"""Two way (mutual nearest neighbor) matcher.

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

from frontend.matcher.matcher_base import MatcherBase


class TwoWayMatcher(MatcherBase):
    """Two way (mutual nearest neighbor) matcher using OpenCV."""

    def match(self,
              descriptors_im1: np.ndarray,
              descriptors_im2: np.ndarray,
              distance_type: str = 'euclidean') -> np.ndarray:
        """Match descriptor vectors.

        Refer to the doc in the parent class for output format.

        Args:
            descriptors_im1: descriptors from image #1, of shape (N1, D).
            descriptors_im2: descriptors from image #2, of shape (N2, D).
            distance_type (optional): the space to compute the distance between
                                      descriptors. Defaults to 'euclidean'.

        Returns:
            Match indices (sorted by confidence), as matrix of shape
                (<min(N1, N2), 2).
        """

        if distance_type == 'euclidean':
            distance_metric = cv.NORM_L2
        elif distance_type == 'hamming':
            distance_metric = cv.NORM_HAMMING
        else:
            raise NotImplementedError(
                'The specified distance type is not implemented')

        if descriptors_im1.size == 0 or descriptors_im2.size == 0:
            return np.array([])

         # we will have to remove NaNs by ourselves
        valid_idx_im1 = np.nonzero(~(np.isnan(descriptors_im1).any(axis=1)))[0]
        valid_idx_im2 = np.nonzero(~(np.isnan(descriptors_im2).any(axis=1)))[0]

        descriptors_1 = descriptors_im1[valid_idx_im1]
        descriptors_2 = descriptors_im2[valid_idx_im2]

        # run OpenCV's matcher
        bf = cv.BFMatcher(normType=distance_metric, crossCheck=True)
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
