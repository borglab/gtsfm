"""
Defining a dummy detector which is to be used in testing.

Authors: Ayush Baid
"""
import numpy as np

from common.image import Image
from frontend.detector.detector_base import DetectorBase


class DummyDetector(DetectorBase):
    """
    A dummy detector which returns random features
    """

    def detect(self, image: Image):
        num_features = 6

        """
        Fill in the columns with random coordinates, scale and optional extra columns

        Constraints:
        1. Coordinates must within the image
        2. scale must be non-negative
        """
        np.random.seed(
            int(1000*np.sum(image.image_array, axis=None) % (2 ^ 32))
        )

        num_columns = 4

        features = np.empty((num_features, num_columns))

        # assign the coordinates
        features[:, :2] = np.random.randint(
            [0, 0], high=[image.image_array.shape[1], image.image_array.shape[0]], size=(num_features, 2))

        # assign the scale
        features[:, 2] = np.random.rand(num_features)

        # assign other dimensions independently
        if num_columns > 3:
            features[:, 3:] = np.random.rand(num_features, num_columns-3)

        return features
