"""A dummy descriptor which is to be used in testing.

Authors: Ayush Baid
"""
import numpy as np

from common.image import Image
from frontend.descriptor.descriptor_base import DescriptorBase


class DummyDescriptor(DescriptorBase):
    """Assigns random vectors as descriptors."""

    def __init__(self):
        super().__init__()

        self.descriptor_length = 15  # length of each descriptor

    def describe(self, image: Image, features: np.ndarray) -> np.ndarray:
        """Assign descriptors to features in an image.

        Output format:
        1. Each input feature point is assigned a descriptor, which is stored
        as a row vector

        Arguments:
            image: the input image.
            features: the features to describe.

        Returns:
            np.ndarray: the descriptors for the input features.
        """
        if features.size == 0:
            return np.array([])

        np.random.seed(int(features[0, 0]))

        return np.random.rand(features.shape[0], self.descriptor_length)
