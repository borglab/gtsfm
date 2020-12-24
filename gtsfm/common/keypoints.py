"""Class to hold coordinates and optional metadata for keypoints, the output of
detections on an image.

Authors: Ayush Baid
"""
import copy
from typing import List, NamedTuple, Optional

import cv2 as cv
import numpy as np

# defaults for OpenCV's Keypoint attributes
OPENCV_DEFAULT_SIZE = 2


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
        coordinates_eq = np.array_equal(self.coordinates, other.coordinates)

        # equality check on scales
        if self.scales is None and other.scales is None:
            scales_eq = True
        elif self.scales is not None and other.scales is not None:
            scales_eq = np.array_equal(self.scales, other.scales)
        else:
            scales_eq = False

        # equality check on responses
        if self.responses is None and other.responses is None:
            responses_eq = True
        elif self.responses is not None and other.responses is not None:
            responses_eq = np.array_equal(self.responses, other.responses)
        else:
            responses_eq = False

        return coordinates_eq and scales_eq and responses_eq

    def __ne__(self, other: object) -> bool:
        """Checks that the other object is not equal to the current object."""
        return not self == other

    def get_top_k(self, k: int) -> "Keypoints":
        """Returns the top keypoints by their response values (or just the
        values from the front in case of missing responses.)
        
        If k keypoints are requested, and only n are available, where n < k,
        then returning n keypoints is the expected behavior.

        Args:
            k: max number of keypoints to return.

        Returns:
            subset of current keypoints.
        """
        if k >= len(self):
            return copy.deepcopy(self)

        if self.responses is None:
            selection_idxs = np.arange(k, dtype=np.uint32)
        else:
            # select the values with top response values
            selection_idxs = np.argpartition(-self.responses, k)[:k]
        return Keypoints(
            coordinates=self.coordinates[selection_idxs],
            scales=self.scales[selection_idxs]
            if self.scales is not None
            else None,
            responses=self.responses[selection_idxs]
            if self.responses is not None
            else None,
        )

    def get_x_coordinates(self) -> np.ndarray:
        """Getter for the x coordinates.

        Returns:
            x coordinates as a vector with the same length as len().
        """
        # TODO: remove function
        return self.coordinates[:, 0]

    def get_y_coordinates(self) -> np.ndarray:
        """Getter for the y coordinates.

        Returns:
            y coordinates as a vector with the same length as len().
        """
        # TODO: remove function
        return self.coordinates[:, 1]

    def cast_to_float(self) -> "Keypoints":
        """Cast all attributes which are numpy arrays to float.

        Returns:
            keypoints with the type-casted attributes.
        """
        return Keypoints(
            coordinates=None
            if self.coordinates is None
            else self.coordinates.astype(np.float32),
            scales=None
            if self.scales is None
            else self.scales.astype(np.float32),
            responses=None
            if self.responses is None
            else self.responses.astype(np.float32),
        )

    def cast_to_opencv_keypoints(self) -> List[cv.KeyPoint]:
        """Cast GTSFM's keypoints to list of OpenCV's keypoints.

        Args:
            keypoints: GTSFM's keypoints.

        Returns:
            List of OpenCV's keypoints with the same information as input keypoints.
        """

        # cast input attributed to floating point numpy arrays.
        keypoints = self.cast_to_float()

        opencv_keypoints = []

        if self.responses is None and keypoints.scales is None:
            for idx in range(len(keypoints)):
                opencv_keypoints.append(
                    cv.KeyPoint(
                        x=keypoints.coordinates[idx, 0],
                        y=keypoints.coordinates[idx, 1],
                        _size=OPENCV_DEFAULT_SIZE,
                    )
                )
        elif keypoints.responses is None:
            for idx in range(len(self)):
                opencv_keypoints.append(
                    cv.KeyPoint(
                        x=keypoints.coordinates[idx, 0],
                        y=keypoints.coordinates[idx, 1],
                        _size=keypoints.scales[idx],
                    )
                )
        elif keypoints.scales is None:
            for idx in range(len(keypoints)):
                opencv_keypoints.append(
                    cv.KeyPoint(
                        x=keypoints.coordinates[idx, 0],
                        y=keypoints.coordinates[idx, 1],
                        _size=OPENCV_DEFAULT_SIZE,
                        _response=keypoints.responses[idx],
                    )
                )
        else:
            for idx in range(len(keypoints)):
                opencv_keypoints.append(
                    cv.KeyPoint(
                        x=keypoints.coordinates[idx, 0],
                        y=keypoints.coordinates[idx, 1],
                        _size=keypoints.scales[idx],
                        _response=keypoints.responses[idx],
                    )
                )

        return opencv_keypoints
