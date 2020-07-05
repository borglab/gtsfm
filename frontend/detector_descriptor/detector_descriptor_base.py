"""
Joint detector and descriptor for the front end.

Authors: Ayush Baid
"""

import abc
from typing import List, Tuple

import dask
import numpy as np

from common.image import Image


class DetectorDescriptorBase(metaclass=abc.ABCMeta):
    """
    Base class for all methods which provide a joint detector-descriptor to work on an image.

    This class serves as a combination of individual detector and descriptor.
    """

    @abc.abstractmethod
    def detect_and_describe(self, image: Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform feature detection as well as their description in a single step.

        Refer to detect() in BaseDetector and describe() in BaseDescriptor 
        for details about the output format.

        Args:
            image (Image): the input image

        Returns:
            Tuple[np.ndarray, np.ndarray]: detected features and their 
                                           descriptions as two numpy arrays
        """

    def create_computation_graph(self, loader_graph: List[dask.delayed]) -> List[dask.delayed]:
        """
        Generates the computation graph for all the entried in the supplied dataset.

        Args:
            loader_graph (List[dask.delayed]): computation graph from loader

        Returns:
            List: delayed dask elements for detect_and_describe()
        """
        return [dask.delayed(self.detect_and_describe)(x) for x in loader_graph]
