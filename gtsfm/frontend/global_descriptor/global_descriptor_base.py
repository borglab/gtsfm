"""Base class for global descriptor implementations.

Authors: John Lambert
"""

import abc

import numpy as np
import torch

from gtsfm.loader.loader_base import BatchTransform, ResizeTransform


class GlobalDescriptorBase:
    """Base class for all the global image descriptors.

    Global image descriptors assign a vector for each input image.
    """

    @abc.abstractmethod
    def describe_batch(self, images: torch.Tensor) -> list[np.ndarray]:
        """Compute global descriptors for a batch of images.

        This is a default, inefficient implementation. Subclasses should override this for true batch processing.

        Args:
            images: A list of input images.

        Returns:
            A list of descriptors, where each is a (D,) numpy array.
        """

        # Default implementation: process one at a time
        # Note: This is inefficient and subclasses should override for true batching

    @abc.abstractmethod
    def get_preprocessing_transforms(self) -> tuple[ResizeTransform, BatchTransform | None]:
        """Return a Resizing Transform and General Batch Transform

        The (required) resizing transform will take in a numpy array and return a resized torch tensor.
        The general batch transform should take a tensor of shape [C, H, W] and return [C, H', W'].

        Returns:
            tuple of (Resize, Optional Batch) Transform functions
        """
