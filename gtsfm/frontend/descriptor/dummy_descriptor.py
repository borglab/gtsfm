"""A dummy descriptor which is to be used in testing.

Authors: Ayush Baid
"""
import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.descriptor.descriptor_base import DescriptorBase


class DummyDescriptor(DescriptorBase):
    """Assigns random vectors as descriptors."""

    def __init__(self):
        super().__init__()

        self.descriptor_length = 15  # length of each descriptor

    def apply(self, image: Image, keypoints: Keypoints) -> np.ndarray:
        """Assign descriptors to detected features in an image, using random
        number generator.

        Arguments:
            image: the input image.
            keypoints: the keypoints to describe, of length N.

        Returns:
            Descriptors for the input features, of shape (N, D) where D is the dimension of each descriptor.
        """
        if len(keypoints) == 0:
            return np.array([])

        np.random.seed(int(keypoints.coordinates[0, 0]))

        return np.random.rand(len(keypoints), self.descriptor_length)
