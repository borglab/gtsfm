"""Class to hold coordinates and optional metadata for keypoints, the output of
detections on an image.

Authors: Ayush Baid
"""
from typing import NamedTuple, Optional

import numpy as np


class Keypoints(NamedTuple):
    coordinates: np.ndarray
    scales: Optional[np.ndarray] = None
    responses: Optional[np.ndarray] = None  # TODO(ayush): enforce the range.
    """Output of detections in an image.

    Coordinate system convention:
        1. The x coordinate denotes the horizontal direction (+ve direction
           towards the right).
        2. The y coordinate denotes the vertical direction (+ve direction
           downwards).
        3. Origin is at the top left corner of the image.

    Args:
        coordinates: the (x, y) coordinates of the features, of shape Nx2.
        scales: optional scale of the detections, of shape N.
        responses: optional respose of the detections, of shape N.
    """

    def __len__(self) -> int:
        """Number of descriptors."""
        return self.coordinates.shape[0]

    def __eq__(self, other: object) -> bool:
        """Checks equality with the other keypoints object."""

        if not isinstance(other, Keypoints):
            return False

        # equality check on coordinates
        coordinates_equality = np.array_equal(
            self.coordinates, other.coordinates)

        # equality check on scales
        if self.scales is None and other.scales is None:
            scale_equality = True
        elif self.scales is not None and other.scales is not None:
            scale_equality = np.array_equal(self.scales, other.scales)
        else:
            scale_equality = False

        # equality check on responses
        if self.responses is None and other.responses is None:
            response_equality = True
        elif self.responses is not None and other.responses is not None:
            response_equality = np.array_equal(self.responses, other.responses)
        else:
            response_equality = False

        return coordinates_equality and scale_equality and response_equality

    def __ne__(self, other: object) -> bool:
        """Checks that the other object is not equal to the current object."""
        return not self == other

    def get_x_coordinates(self) -> np.ndarray:
        """Getter for the x coordinates.

        Returns:
            x coordinates as a vector with the same length as len().
        """
        return self.coordinates[:, 0]

    def get_y_coordinates(self) -> np.ndarray:
        """Getter for the y coordinates.

        Returns:
            y coordinates as a vector with the same length as len().
        """
        return self.coordinates[:, 1]
