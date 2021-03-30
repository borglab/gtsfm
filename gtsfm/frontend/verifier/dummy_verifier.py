"""Dummy matcher which produces random results.

Authors: Ayush Baid
"""
from typing import Optional, Tuple

import numpy as np
from gtsam import Cal3Bundler, Point3, Rot3, Unit3

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.verifier_base import VerifierBase

# constant to be used for keeping random seed in int range.
UINT32_MAX = 2 ** 32


class DummyVerifier(VerifierBase):
    """A dummy verifier which produces random results"""

    def __init__(self):
        super().__init__(min_pts_e_matrix=5, min_pts_f_matrix=8)

    def verify(
        self,
        keypoints_i1: Keypoints,  # pylint: disable=unused-argument
        keypoints_i2: Keypoints,  # pylint: disable=unused-argument
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,  # pylint: disable=unused-argument
        camera_intrinsics_i2: Cal3Bundler,  # pylint: disable=unused-argument
        use_intrinsics_in_verification: bool = False,  # pylint: disable=unused-argument
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Performs verification of correspondences between two images to recover the relative pose and indices of
        verified correspondences.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.
            use_intrinsics_in_verification (optional): Flag to perform keypoint normalization and compute the essential
                                                       matrix instead of fundamental matrix. This should be preferred
                                                       when the exact intrinsics are known as opposed to approximating
                                                       them from exif data. Defaults to False.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3. These are subset of match_indices.
        """
        v_inlier_idxs = np.array([], dtype=np.uint32)

        # check if we don't have the minimum number of points
        if match_indices.shape[0] < self.min_pts_e_matrix:
            return None, None, v_inlier_idxs

        # set a random seed using descriptor data for repeatability
        np.random.seed(int(1000 * (match_indices[0, 0] + match_indices[0, 1]) % (UINT32_MAX)))

        # get the number of entries in the input
        num_matches = match_indices.shape[0]

        # get the number of verified_pts we will output
        num_verifier_pts = np.random.randint(low=0, high=num_matches)

        # randomly sample the indices for matches which will be verified
        v_inlier_idxs = np.random.choice(num_matches, num_verifier_pts, replace=False).astype(np.uint32)

        # use a random 3x3 matrix if the number of verified points are less that
        if num_verifier_pts >= self.min_pts:
            # generate random rotation and translation for essential matrix
            rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
            i2Ri1 = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
            i2Ti1 = Point3(np.random.uniform(low=-1.0, high=1.0, size=(3,)))

            return i2Ri1, Unit3(i2Ti1), match_indices[v_inlier_idxs]
        else:
            return None, None, match_indices[v_inlier_idxs]
