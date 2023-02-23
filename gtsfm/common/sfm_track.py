"""Utilities to generate and store tracks. Uses the Union-Find algorithm, with image ID and keypoint index for that
image as the unique keys.

A track is defined as a 2d measurement of a single 3d landmark seen in multiple different images.

References:
1. P. Moulon, P. Monasse. Unordered Feature Tracking Made Fast and Easy, 2012, HAL Archives.
   https://hal-enpc.archives-ouvertes.fr/hal-00769267/file/moulon_monasse_featureTracking_CVMP12.pdf

Authors: Ayush Baid, Sushmita Warrier, John Lambert
"""
from typing import List, NamedTuple, Set

import numpy as np


class SfmMeasurement(NamedTuple):
    """2d measurements (points in images)."""

    i: int  # camera index
    uv: np.ndarray  # 2d measurement

    def __eq__(self, other: object) -> bool:
        """Checks equality with the other object."""
        if not isinstance(other, SfmMeasurement):
            return False

        if self.i != other.i:
            return False

        return np.allclose(self.uv, other.uv)

    def __ne__(self, other: object) -> bool:
        """Checks inequality with the other object."""
        return not self == other


class SfmTrack2d(NamedTuple):
    """Track containing 2D measurements associated with a single 3D point.

    Note: Equivalent to gtsam.SfmTrack, but without the 3d measurement. This class holds data temporarily before 3D
          point is initialized.
    """

    measurements: List[SfmMeasurement]

    def number_measurements(self) -> int:
        """Returns the number of measurements."""
        return len(self.measurements)

    def measurement(self, idx: int) -> SfmMeasurement:
        """Getter for measurement at a particular index.

        Args:
            idx: index to fetch.

        Returns:
            measurement at the requested index.
        """
        return self.measurements[idx]

    def select_subset(self, idxs: List[int]) -> "SfmTrack2d":
        """Generates a new track with the subset of measurements.

        Returns:
            Track with the subset of measurements.
        """
        inlier_measurements = [self.measurements[j] for j in idxs]
        return SfmTrack2d(inlier_measurements)

    def select_for_cameras(self, camera_idxs: Set[int]) -> "SfmTrack2d":
        """Generates a new track with only those measurements which are in camera_idxs.

        Unlike `select_subset`, this method does not require all camera_idxs to have a measurement in this track.

        Returns:
            Track with the subset of measurements.
        """
        measurements = [m for m in self.measurements if m.i in camera_idxs]
        return SfmTrack2d(measurements)

    def __eq__(self, other: object) -> bool:
        """Checks equality with the other object."""

        # check object type
        if not isinstance(other, SfmTrack2d):
            return False

        # check number of measurements
        if len(self.measurements) != len(other.measurements):
            return False

        # check the individual measurements (order insensitive)
        # inefficient implementation but wont be used a lot
        for measurement in self.measurements:
            if measurement not in other.measurements:
                return False

        return True

    def __ne__(self, other: object) -> bool:
        """Checks inequality with the other object."""
        return not self == other

    def validate_unique_cameras(self) -> bool:
        """Validates the track by checking that no two measurements are from the same camera.

        Returns:
            boolean result of the validation.
        """
        track_cam_idxs = [measurement.i for measurement in self.measurements]
        return len(set(track_cam_idxs)) == len(track_cam_idxs)
