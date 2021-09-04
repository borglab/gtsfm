"""SIFT Detector-Descriptor from Colmap.

Ref: 

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np
import pycolmap

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


class SIFTColmap(DetectorDescriptorBase):
    """SIFT detector-descriptor using OpenCV's implementation."""

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for details about the output format.

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """
        vlfeat_keypoints, responses, descriptors = pycolmap.extract_sift(image.value_array)

        # sort the features and descriptors by the score
        # (need to sort here as we need the sorting order for descriptors)
        sort_idx = np.argsort(-responses)[: self.max_keypoints]

        gtsfm_keypoints = Keypoints(
            coordinates=vlfeat_keypoints[sort_idx, :2],
            scales=vlfeat_keypoints[sort_idx, 2],
            responses=responses[sort_idx],
        )

        return gtsfm_keypoints, descriptors[sort_idx]
