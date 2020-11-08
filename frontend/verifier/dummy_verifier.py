"""
Dummy matcher which produces random results.

Authors: Ayush Baid
"""
from typing import Tuple, Optional

import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, EssentialMatrix, Point3, Rot3

from frontend.verifier.verifier_base import VerifierBase


class DummyVerifier(VerifierBase):
    """A dummy verifier which produces random results"""

    def __init__(self):
        super().__init__(min_pts=8)

    def verify_with_exact_intrinsics(
        self,
        features_im1: np.ndarray,
        features_im2: np.ndarray,
        match_indices: np.ndarray,
        camera_instrinsics_im1: Cal3Bundler,
        camera_instrinsics_im2: Cal3Bundler,
    ) -> Tuple[Optional[EssentialMatrix], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is prefered when camera intrinsics are known. The
        feature coordinates are normalized and the essential matrix is directly
        estimated.

        Args:
            features_im1: detected features in image #1, of shape (N1, 2+).
            features_im2: detected features in image #2, of shape (N2, 2+).
            match_indices: matches as indices of features from both images, of
                           shape (N3, 2), where N3 <= min(N1, N2).
            camera_instrinsics_im1: intrinsics for image #1.
            camera_instrinsics_im2: intrinsics for image #2.

        Returns:
            Estimated essential matrix im2_E_im1, or None if it cannot be 
                estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3.
                These indices are subset of match_indices.
        """
        essential_matrix = None
        verified_indices = np.array([], dtype=np.uint32)

        # check if we dont have the minimum number of points
        if match_indices.size <= self.min_pts:
            return essential_matrix, verified_indices

        # set a random seed using descriptor data for repeatibility
        np.random.seed(
            int(1000*(match_indices[0, 0] +
                      match_indices[0, 1]) % (2 ^ 32))
        )

        # get the number of entries in the input
        num_matches = match_indices.shape[0]

        # get the number of verified_pts we will output
        num_verifier_pts = np.random.randint(
            low=0, high=num_matches)

        # randomly sample the indices for matches which will be verified
        verified_matches = np.random.choice(
            num_matches, num_verifier_pts, replace=False).astype(np.uint32)

        # use a random 3x3 matrix if the number of verified points are less that
        if num_verifier_pts >= self.min_pts:
            # generate random rotation and translation for essential matrix
            rotation_angles = np.random.uniform(
                low=0.0, high=2*np.pi, size=(3,))
            im2_R_im1 = Rot3.RzRyRx(
                rotation_angles[0], rotation_angles[1], rotation_angles[2])
            im2_t_im1 = Point3(np.random.uniform(
                low=-1.0, high=1.0, size=(3, )))

            essential_matrix = EssentialMatrix(im2_R_im1, im2_t_im1)

        return essential_matrix, match_indices[verified_matches]

    def verify_with_approximate_intrinsics(
        self,
        features_im1: np.ndarray,
        features_im2: np.ndarray,
        match_indices: np.ndarray,
        camera_instrinsics_im1: Cal3Bundler,
        camera_instrinsics_im2: Cal3Bundler,
    ) -> Tuple[Optional[EssentialMatrix], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is prefered when camera intrinsics are approximate
        (i.e from image size/exif). The feature coordinates are used to compute
        the fundamental matrix, which is then converted to the essential matrix.

        Args:
            features_im1: detected features in image #1, of shape (N1, 2+).
            features_im2: detected features in image #2, of shape (N2, 2+).
            match_indices: matches as indices of features from both images, of
                           shape (N3, 2), where N3 <= min(N1, N2).
            camera_instrinsics_im1: intrinsics for image #1.
            camera_instrinsics_im2: intrinsics for image #2.

        Returns:
            Estimated essential matrix im2_E_im1, or None if it cannot be
                estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3.
                These indices are subset of match_indices.
        """
        # call the function for exact intrinsics as this is a dummy verifier.
        self.verify_with_exact_intrinsics(
            features_im1, features_im2, match_indices, camera_instrinsics_im1, camera_instrinsics_im2)
