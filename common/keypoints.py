"""Class to hold coordinates and optional metadata for keypoints, the output of
detections on an image.

Authors: Ayush Baid
"""

from typing import NamedTuple, Optional

import numpy as np


class Keypoints(NamedTuple):
    coordinates: np.ndarray
    scale: Optional[np.ndarray] = None
    response: Optional[np.ndarray] = None
    """Output of detections in an image.

    Args:
        coordinates: the (x, y) coordinates of the features, of shape Nx2.
        scale: optional scale of the detections, of shape N.
        response: optional respose of the detections, of shape N.
    """

    def __len__(self) -> int:
        """Number of descriptors."""
        return self.coordinates.shape[0]

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
