"""Base class for the description stage of the frontend.

Authors: Ayush Baid
"""
import abc

import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints


class DescriptorBase(metaclass=abc.ABCMeta):
    """Base class for all the feature descriptors.

    Feature descriptors assign a vector for each input point.
    """

    @abc.abstractmethod
    def describe(self, image: Image, keypoints: Keypoints) -> np.ndarray:
        """Assign descriptors to detected features in an image.

        Arguments:
            image: the input image.
            keypoints: the keypoints to describe, of length N.

        Returns:
            Descriptors for the input features, of shape (N, D) where D is the dimension of each descriptor.
        """
