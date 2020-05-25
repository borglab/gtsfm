""" 
Base class for the D (detector) stage of the frontend

Authors: Ayush Baid
"""

import abc
from typing import List

import dask
import numpy as np


class DetectorBase:
    """
    Base class for all the feature detectors
    """

    def create_computation_graph(self, loader_graph) -> List:
        """
        Generates the computation graph for all the entried in the supplied dataset

        Args:
            loader_graph [(type)]: computation graph from loader

        Returns:
            List: delayed dask elements
        """
        # TODO(ayush): check is this is the right thing to do
        return [dask.delayed(self.detect)(x) for x in loader_graph]

    @abc.abstractmethod
    def detect(self, image: np.array) -> np.array:
        """
        Detect the features in an image

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
            image (np.array): the input RGB image as a 3D numpy array

        Returns:
            features (np.array): detected features as a numpy array
        """
