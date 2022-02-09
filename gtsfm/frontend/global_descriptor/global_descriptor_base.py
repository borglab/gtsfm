"""Base class for global descriptor implementations.

Authors: John Lambert
"""

import abc

import numpy as np

from gtsfm.common.image import Image


class GlobalDescriptorBase:

    @abc.abstractmethod
    def describe(self, image: Image) -> np.ndarray:
        """Compute the global descriptor for a single image query."""
