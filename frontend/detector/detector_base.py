""" 
Base class for the Detection stage of the frontend.

Authors: Ayush Baid
"""

import abc
from typing import List

import dask
import numpy as np

from common.image import Image


class DetectorBase(metaclass=abc.ABCMeta):
    """Base class for all the feature detectors."""

    @abc.abstractmethod
    def detect(self, image: Image) -> np.ndarray:
        """
        Detect the features in an image.

        Coordinate system convention:
        1. The x coordinate denotes the horizontal dfirection (+ve direction towards the right)
        2. The y coordinate denotes the vertical direction (+ve direction downwards)
        3. Origin is at the top left corner of the image

        Output format:
        1. The first two columns are the x and y coordinate respectively. They are compulsory.
        2. The third optional column contains the scale information of the features.
        3. Any extra column might not be considered downstream
        4. If applicable, the keypoints should be sorted in decreasing order of score/confidence

        Arguments:
            image (Image): the input RGB image as a 3D numpy array

        Returns:
            features (np.ndarray[float]): detected features as a numpy array
        """

    def create_computation_graph(self, loader_graph: List[dask.delayed]) -> List[dask.delayed]:
        """
        Generates the computation graph for all the entried in the supplied dataset.

        Args:
            loader_graph (List[dask.delayed]): computation graph from loader

        Returns:
            List[dask.delayed]: delayed dask elements
        """
        return [dask.delayed(self.detect)(x) for x in loader_graph]
