"""Base class for global descriptor implementations.

Authors: John Lambert
"""

import abc

import numpy as np

from typing import List
from gtsfm.common.image import Image


class GlobalDescriptorBase:
    """Base class for all the global image descriptors.

    Global image descriptors assign a vector for each input image.
    """

    @abc.abstractmethod
    def describe(self, image: Image) -> np.ndarray:
        """Compute the global descriptor for a single image query.

        Args:
            image: input image.

        Returns:
            img_desc: array of shape (D,) representing global image descriptor.
        """

    def describe_batch(self, images: List[Image]) -> List[np.ndarray]:
        """Compute global descriptors for a batch of images.

        This is a default, inefficient implementation. Subclasses should override this for true batch processing.

        Args:
            images: A list of input images.

        Returns:
            A list of descriptors, where each is a (D,) numpy array.
        """
        return [self.describe(image) for image in images]
