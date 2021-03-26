"""Class to construct joint matcher-verifier from standalone matcher and verifier.

Authors: Ayush Baid
"""
from typing import Optional, Tuple

import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.matcher_verifier.matcher_verifier_base import MatcherVerifierBase
from gtsfm.frontend.verifier.verifier_base import VerifierBase


class CombinationMatcherVerifier(MatcherVerifierBase):
    """Combines standalone matcher and verifier."""

    def __init__(self, matcher: MatcherBase, verifier: VerifierBase) -> None:
        """Initializes the combined class.

        Args:
            matcher: standalone matcher.
            verifier: standalone verifier.
        """
        self._matcher = matcher
        self._verifier = verifier

    def match_and_verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
        use_intrinsics_in_verification: bool = False,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Matches the descriptors to generate putative correspondences, and then verifies them to estimate the
        essential matrix and verified correspondences.

        Args:
            keypoints_i1: detected features in image #i1, of length N1.
            keypoints_i2: detected features in image #i2, of length N2.
            descriptors_i1: descriptors for keypoints_i1.
            descriptors_i2: descriptors for keypoints_i2.
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.
            use_intrinsics_in_verification (optional): Flag to perform keypoint normalization and compute the essential
                                                       matrix instead of fundamental matrix. This should be preferred
                                                       when the exact intrinsics are known as opposed to approximating
                                                       them from exif data. Defaults to False.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= min(N1, N2).
        """

        match_corr_idx = self._matcher.match(descriptors_i1, descriptors_i2)
        return self._verifier.verify(
            keypoints_i1,
            keypoints_i2,
            match_corr_idx,
            camera_intrinsics_i1,
            camera_intrinsics_i2,
            use_intrinsics_in_verification,
        )
