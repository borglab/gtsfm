"""A dummy detector which is to be used in testing.

Authors: Ayush Baid
"""
import numpy as np

from common.image import Image
from common.keypoints import Keypoints
from frontend.detector.detector_base import DetectorBase


class DummyDetector(DetectorBase):
    """Assigns random features to an input image."""

    def detect(self, image: Image) -> Keypoints:
        """Detect the features in an image by using random numbers.

        Coordinate system convention:
        1. The x coordinate denotes the horizontal direction (+ve direction
           towards the right).
        2. The y coordinate denotes the vertical direction (+ve direction
           downwards).
        3. Origin is at the top left corner of the image.

        Output format:
        1. If applicable, the keypoints should be sorted in decreasing order of
           score/confidence.

        Args:
            image: input image.

        Returns:
            detected keypoints, with maximum length of max_keypoints.
        """

        np.random.seed(
            int(1000*np.sum(image.value_array, axis=None) % (2 ^ 32))
        )

        num_detections = np.random.randint(0, high=15, size=(1)).item()

        # assign the coordinates
        coordinates = np.random.randint(
            low=[0, 0],
            high=[image.value_array.shape[1], image.value_array.shape[0]],
            size=(num_detections, 2))

        # assign the scale
        scale = np.random.rand(num_detections)

        response = np.random.rand(num_detections)

        return Keypoints(coordinates, scale, response)
