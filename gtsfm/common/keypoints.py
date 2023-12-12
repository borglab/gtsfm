"""Class to hold coordinates and optional metadata for keypoints, the output of detections on an image.

Authors: Ayush Baid
"""
import copy
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

# defaults for OpenCV's Keypoint attributes
OPENCV_DEFAULT_SIZE = 2


class Keypoints:
    """Output of detections in an image.

    Coordinate system convention:
        1. The x coordinate denotes the horizontal direction (+ve direction towards the right).
        2. The y coordinate denotes the vertical direction (+ve direction downwards).
        3. Origin is at the top left corner of the image.

    Note: the keypoints class *should not* be implemented as NamedTuple because dask treats NamedTuple as a tuple and
    tries to estimate the size by randomly sampling elements (number computed using len()). len() is used for the
    number of points and not the number of attributes.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        scales: Optional[np.ndarray] = None,
        oris: Optional[np.ndarray] = None,
        responses: Optional[np.ndarray] = None,
    ):
        """Initializes the attributes.

        Args:
            coordinates: The (x, y) coordinates of the features, of shape Nx2.
            scales: Optional scale of the detections, of shape N.
            responses: Optional confidences/responses for each detection, of shape N.
        """
        self.coordinates = coordinates
        self.scales = scales
        self.oris = oris
        self.responses = responses  # TODO(ayush): enforce the range.

    def __len__(self) -> int:
        """Number of descriptors."""
        return self.coordinates.shape[0]

    def __sizeof__(self) -> int:
        """Functionality required by Dask to avoid warnings."""
        return (
            super().__sizeof__()
            + self.coordinates.__sizeof__()
            + self.scales.__sizeof__()
            + self.responses.__sizeof__()
        )

    def __eq__(self, other: object) -> bool:
        """Checks equality with the other keypoints object."""

        if not isinstance(other, Keypoints):
            return False

        # Equality check on coordinates.
        coordinates_eq = np.array_equal(self.coordinates, other.coordinates)

        # Equality check on scales.
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

    def get_top_k(self, k: int) -> Tuple["Keypoints", np.ndarray]:
        """Returns the top keypoints by their response values (or just the values from the front in case of missing
        responses.)

        If k keypoints are requested, and only n < k are available, then returning n keypoints is the expected behavior.

        Args:
            k: Maximum number of keypoints to return.

        Returns:
            Subset of current keypoints.
        """
        if k >= len(self):
            return copy.deepcopy(self), np.arange(self.__len__())

        if self.responses is None:
            selection_idxs = np.arange(k, dtype=np.uint32)
        else:
            # select the values with top response values
            selection_idxs = np.argpartition(-self.responses, k)[:k]

        return self.extract_indices(selection_idxs), selection_idxs

    def filter_by_mask(self, mask: np.ndarray) -> Tuple["Keypoints", np.ndarray]:
        """Filter features with respect to a binary mask of the image.

        Args:
            mask: (H, W) array of 0's and 1's corresponding to valid portions of the original image.
            keypoints: Detected keypoints with length M.
            descriptors: (M, D) array of descriptors D is the dimension of each descriptor.

        Returns:
            N <= M keypoints, and their corresponding desciptors as an (N, D) array, such that their (rounded)
                coordinates corresponded to a 1 in the input mask array.
        """
        rounded_coordinates = np.round(self.coordinates).astype(int)
        valid_idxs = np.flatnonzero(mask[rounded_coordinates[:, 1], rounded_coordinates[:, 0]] == 1)

        return self.extract_indices(valid_idxs), valid_idxs

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
            Keypoints with the type-casted attributes.
        """
        return Keypoints(
            coordinates=None if self.coordinates is None else self.coordinates.astype(np.float32),
            scales=None if self.scales is None else self.scales.astype(np.float32),
            responses=None if self.responses is None else self.responses.astype(np.float32),
        )

    def cast_to_opencv_keypoints(self) -> List[cv.KeyPoint]:
        """Cast GTSFM's keypoints to list of OpenCV's keypoints.

        Args:
            keypoints: GTSFM's keypoints.

        Returns:
            List of OpenCV's keypoints with the same information as input keypoints.
        """

        # Cast input attributed to floating point numpy arrays.
        keypoints = self.cast_to_float()

        opencv_keypoints = []

        if self.responses is None and keypoints.scales is None:
            for idx in range(len(keypoints)):
                opencv_keypoints.append(
                    cv.KeyPoint(
                        x=keypoints.coordinates[idx, 0],
                        y=keypoints.coordinates[idx, 1],
                        size=OPENCV_DEFAULT_SIZE,
                    )
                )
        elif keypoints.responses is None:
            for idx in range(len(self)):
                opencv_keypoints.append(
                    cv.KeyPoint(
                        x=keypoints.coordinates[idx, 0],
                        y=keypoints.coordinates[idx, 1],
                        size=keypoints.scales[idx],
                    )
                )
        elif keypoints.scales is None:
            for idx in range(len(keypoints)):
                opencv_keypoints.append(
                    cv.KeyPoint(
                        x=keypoints.coordinates[idx, 0],
                        y=keypoints.coordinates[idx, 1],
                        size=OPENCV_DEFAULT_SIZE,
                        response=keypoints.responses[idx],
                    )
                )
        else:
            for idx in range(len(keypoints)):
                opencv_keypoints.append(
                    cv.KeyPoint(
                        x=keypoints.coordinates[idx, 0],
                        y=keypoints.coordinates[idx, 1],
                        size=keypoints.scales[idx],
                        response=keypoints.responses[idx],
                    )
                )

        return opencv_keypoints

    def extract_indices(self, indices: np.ndarray) -> "Keypoints":
        """Form subset with the given indices.

        Args:
            indices: Indices to extract, as a 1-D vector.

        Returns:
            Subset of data at the given indices.
        """
        if indices.size == 0:
            return Keypoints(coordinates=np.zeros(shape=(0, 2)))

        return Keypoints(
            self.coordinates[indices],
            None if self.scales is None else self.scales[indices],
            None if self.oris is None else self.scales[indices],
            None if self.responses is None else self.responses[indices],
        )
