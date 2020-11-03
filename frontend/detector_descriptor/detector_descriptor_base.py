"""Joint detector and descriptor for the front end.

Authors: Ayush Baid
"""

import abc
from typing import List, Tuple

import dask
import numpy as np
from dask.delayed import Delayed

from common.image import Image


class DetectorDescriptorBase(metaclass=abc.ABCMeta):
    """Base class for all methods which provide a joint detector-descriptor to
    work on an image.

    This class serves as a combination of individual detector and descriptor.
    """

    def __init__(self):
        self.max_features = 5000

    @abc.abstractmethod
    def detect_and_describe(self,
                            image: Image) -> Tuple[np.ndarray, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for
        details about the output format.

        Args:
            image: the input image.

        Returns:
            detected features as a numpy array of shape (N, 2+).
            corr. descriptors for the features, as (N, x) sized matrix.
        """

    def create_computation_graph(self,
                                 loader_graph: List[Delayed]
                                 ) -> List[Delayed]:
        """
        Generates the computation graph for all the entried in the supplied dataset.

        Args:
            loader_graph: computation graph from loader.

        Returns:
            delayed dask elements for joint detection-description.
        """
        return [dask.delayed(self.detect_and_describe)(x) for x in loader_graph]
