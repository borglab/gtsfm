"""A dummy detector which is to be used in testing.

Authors: Ayush Baid
"""
import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector.detector_base import DetectorBase


class DummyDetector(DetectorBase):
    """Assigns random features to an input image."""

    def detect(self, image: Image) -> Keypoints:
        """Detect the features in an image by using random numbers.

        Args:
            image: input image.

        Returns:
            detected keypoints, with maximum length of max_keypoints.
        """

        np.random.seed(int(1000 * np.sum(image.value_array, axis=None) % (2 ^ 32)))

        num_detections = np.random.randint(0, high=15, size=(1)).item()

        # assign the coordinates
        coordinates = np.random.randint(
            low=[0, 0],
            high=[image.value_array.shape[1], image.value_array.shape[0]],
            size=(num_detections, 2),
        )

        # assign the scale
        scales = np.random.rand(num_detections)

        # assing responses
        responses = np.random.rand(num_detections)

        return Keypoints(coordinates, scales, responses)
