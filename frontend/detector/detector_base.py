"""Base class for the Detection stage of the frontend.

Authors: Ayush Baid
"""

import abc
from typing import List

import dask
import numpy as np
from dask.delayed import Delayed

from common.image import Image


class DetectorBase(metaclass=abc.ABCMeta):
    """Base class for all the feature detectors."""

    def __init__(self, max_features: int = 5000) -> None:
        """Initialize the detector.

        Args:
            max_features: Maximum number of features to detect. Defaults to
                          5000.
        """
        self.max_features = max_features

    @abc.abstractmethod
    def detect(self, image: Image) -> np.ndarray:
        """Detect the features in an image.

        Coordinate system convention:
        1. The x coordinate denotes the horizontal direction (+ve direction
           towards the right)
        2. The y coordinate denotes the vertical direction (+ve direction
           downwards)
        3. Origin is at the top left corner of the image

        Output format:
        1. The first two columns are the x and y coordinate respectively. They 
           are compulsory.
        2. The third optional column contains the scale information of the features.
        3. The fourth optional column contains the confidence of the feature. 
           TODO: what is the normalization/scale. 
           Superpoint uses 0-1 scale (high is better and we will go ahead with it)
        4. Any extra column might not be considered downstream
        5. If applicable, the keypoints should be sorted in decreasing order of
           score/confidence

        Args:
            image: input image.

        Returns:
            detected features.
        """

    def create_computation_graph(self,
                                 loader_graph: List[Delayed]
                                 ) -> List[Delayed]:
        """Generates the computation graph for all the entries in the
        loader graph.

        Args:
            loader_graph (List[Delayed]): computation graph from loader

        Returns:
            List[Delayed]: delayed dask elements
        """
        return [dask.delayed(self.detect)(x) for x in loader_graph]
