"""Base class for global descriptor implementations.

Authors: John Lambert
"""

import abc

import numpy as np
import torch

from typing import List, Optional, Callable
from gtsfm.common.image import Image


class GlobalDescriptorBase:
    """Base class for all the global image descriptors.

    Global image descriptors assign a vector for each input image.
    """

    @abc.abstractmethod
    def describe(self, image: Image) -> np.ndarray:
        """Compute the global descriptor for a single image query.

        Args:
            image: input GTSFM Image Object

        Returns:
            img_desc: array of shape (D,) representing global image descriptor.
        """

    def describe_batch(self, images: torch.Tensor) -> List[np.ndarray]:
        """Compute global descriptors for a batch of images.

        This is a default, inefficient implementation. Subclasses should override this for true batch processing.

        Args:
            images: A list of input images.

        Returns:
            A list of descriptors, where each is a (D,) numpy array.
        """
        
        # Default implementation: process one at a time
        # Note: This is inefficient and subclasses should override for true batching
        
        descriptors = []
        for i in range(images.shape[0]):
            # Create temporary Image object for each tensor
            image_array = images[i].cpu().numpy()
            temp_image = Image(value_array=image_array)
            descriptors.append(self.describe(temp_image))
        return descriptors

    def get_preprocessing_transform(self) -> Optional[Callable]:
        """Return preprocessing transform to apply to image tensors before inference.
        
        This transform will be applied by the loader when creating batches.
        The transform should take a tensor of shape [C, H, W] and return [C, H', W'].
        
        Returns:
            Optional transform function, or None if no preprocessing needed.
        """
        return None  # Default: no transform needed
