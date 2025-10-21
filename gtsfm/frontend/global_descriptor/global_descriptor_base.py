"""Base class for global descriptor implementations.

Authors: John Lambert
"""

import abc

import numpy as np

from typing import Optional, Callable
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
    
    def get_preprocessing_transform(self) -> Optional[Callable[[Image], Image]]:
        """Return a function to preprocess an image before descriptor inference.
        
        This transform will be applied by the loader.
        
        Returns:
            Optional transform function, or None if no preprocessing is needed.
        """
        return None  # Default: no transform needed
