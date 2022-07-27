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
    """Base class for matchers that accept an image pair, and directly generate keypoint matches.

    Note: these matchers do NOT use descriptors as input.
    """

    @abc.abstractmethod
    def match(
        self,
        image_i1: Image,
        image_i2: Image,
    ) -> Tuple[Keypoints, Keypoints]:
        """Identify feature matches across two images.

        Args:
            image_i1: first input image of pair.
            image_i2: second input image of pair.

        Returns:
            Keypoints from image 1 (N keypoints will exist).
            Corresponding keypoints from image 2 (there will also be N keypoints). These represent feature matches.
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
            Delayed dask task for N keypoints from image 1.
            Delayed dask task for N keypoints from image 2.
        """
        return dask.delayed(self.match, nout=2)(image_i1=image_i1, image_i2=image_i2)
