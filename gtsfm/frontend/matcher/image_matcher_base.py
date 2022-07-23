"""Base class for front end matchers that directly identify image correspondences, without using explicit descriptors.

Authors: John Lambert
"""
import abc
from typing import Tuple

import dask
from dask.delayed import Delayed

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints


class ImageMatcherBase(metaclass=abc.ABCMeta):
    """Base class for all matchers.

    Matchers work on a pair of descriptors and match them by their distance.
    """

    @abc.abstractmethod
    def match(
        self,
        image_i1: Image,
        image_i2: Image,
    ) -> Tuple[Keypoints, Keypoints]:
        """Identify feature matches across two images.

        If the results are in the cache, they are fetched and returned. Otherwise, the `match()` of the
        underlying object's API is called and the results are cached.

        Args:
            image_i1: first input image of pair.
            image_i2: second input image of pair.

        Returns:
            Keypoints object with N keypoints.
            Keypoints object with N corresponding keypoints (representing matches).
        """

    def create_computation_graph(
        self,
        image_i1,
        image_i2,
    ) -> Tuple[Delayed, Delayed]:
        """Generates computation graph to directly identify feature matches across two images.

        Args:
            image_i1: first input image of pair.
            image_i2: second input image of pair.

        Returns:
            Delayed dask tasks for matching for input camera pairs, a tuple of Keypoints.
        """
        return dask.delayed(self.match, nout=2)(image_i1=image_i1, image_i2=image_i2)
