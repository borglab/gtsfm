"""Dummy matcher which produces random results.

Authors: Ayush Baid
"""
from typing import Optional, Tuple

import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Point3, Rot3, Unit3

from common.keypoints import Keypoints
from frontend.verifier.verifier_base import VerifierBase

# constant to be used for keeping random seed in int range.
UINT32_MAX = 2 ** 32


class DummyVerifier(VerifierBase):
    """A dummy verifier which produces random results"""

    def __init__(self):
        super().__init__(min_pts=8)

    def verify_with_exact_intrinsics(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is preferred when camera intrinsics are approximate
        (i.e from image size/exif). The feature coordinates are used to compute
        the fundamental matrix, which is then converted to the essential matrix.

        Args:
            keypoints_im1: detected features in image #i1.
            keypoints_im2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of
                           shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_im1: intrinsics for image #i1.
            camera_intrinsics_im2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3.
                These indices are subset of match_indices.
        """
        v_inlier_idxs = np.array([], dtype=np.uint32)

        # check if we don't have the minimum number of points
        if match_indices.shape[0] < self.min_pts:
            return None, None, v_inlier_idxs

        # set a random seed using descriptor data for repeatability
        np.random.seed(
            int(1000*(match_indices[0, 0] +
                      match_indices[0, 1]) % (UINT32_MAX))
        )

        # get the number of entries in the input
        num_matches = match_indices.shape[0]

        # get the number of verified_pts we will output
        num_verifier_pts = np.random.randint(
            low=0, high=num_matches)

        # randomly sample the indices for matches which will be verified
        v_inlier_idxs = np.random.choice(
            num_matches, num_verifier_pts, replace=False).astype(np.uint32)

        # use a random 3x3 matrix if the number of verified points are less that
        if num_verifier_pts >= self.min_pts:
            # generate random rotation and translation for essential matrix
            rotation_angles = np.random.uniform(
                low=0.0, high=2*np.pi, size=(3,))
            i2Ri1 = Rot3.RzRyRx(
                rotation_angles[0], rotation_angles[1], rotation_angles[2])
            i2Ti1 = Point3(np.random.uniform(
                low=-1.0, high=1.0, size=(3, )))

            return i2Ri1, Unit3(i2Ti1), match_indices[v_inlier_idxs]
        else:
            return None, None, match_indices[v_inlier_idxs]

    def verify_with_approximate_intrinsics(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is preferred when camera intrinsics are approximate
        (i.e from image size/exif). The feature coordinates are used to compute
        the fundamental matrix, which is then converted to the essential matrix.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of
                           shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3.
                These indices are subset of match_indices.
        """
        # call the function for exact intrinsics as this is a dummy verifier.
        return self.verify_with_exact_intrinsics(
            keypoints_i1, keypoints_i2, match_indices,
            camera_intrinsics_i1, camera_intrinsics_i2)
