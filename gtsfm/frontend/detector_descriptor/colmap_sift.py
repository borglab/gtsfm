"""Colmap's SIFT detector-descriptor.

This detector-descriptor uses a third-party implementation.

Reference: https://github.com/colmap/pycolmap#sift-feature-extraction

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np
import pycolmap

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


class ColmapSIFTDetectorDescriptor(DetectorDescriptorBase):
    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        grayscale_vals: np.ndarray[np.uint8] = image_utils.rgb_to_gray_cv(image).value_array

        keypoints_pycolmap, keypoint_scores, descriptors = pycolmap.extract_sift(
            grayscale_vals.astype(np.float64) / 255
        )
        return (
            Keypoints(
                coordinates=keypoints_pycolmap[:, :2], scales=keypoints_pycolmap[:, 2], responses=keypoint_scores
            ),
            descriptors,
        )
