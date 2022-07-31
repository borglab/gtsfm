""" LoFTR image matcher.

The network was proposed in "LoFTR: Detector-Free Local Feature Matching with Transformers" and we wrap Kornia's API
to use the matcher in GTSFM.

References:
- https://arxiv.org/pdf/2104.00680v1.pdf
- https://kornia.readthedocs.io/en/latest/_modules/kornia/feature/loftr/loftr.html

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np
import torch
from kornia.feature import LoFTR as LoFTRKornia

import gtsfm.utils.images as image_utils
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints

KEYPOINTS_I1_COORDINATES_KEY = "keypoints0"
KEYPOINTS_I2_COORDINATES_KEY = "keypoints1"


class LOFTR(ImageMatcherBase):
    """LOFTR image matcher."""

    def __init__(self, use_outdoor_model: bool = True) -> None:
        super().__init__()
        self._model_type = "outdoor" if use_outdoor_model else "indoor"

    def match(self, image_i1: Image, image_i2: Image) -> Tuple[Keypoints, Keypoints]:
        """Identify feature matches across two images.

        Note: the matcher will run out of memory for large image sizes

        Args:
            image_i1: first input image of pair.
            image_i2: second input image of pair.

        Returns:
            Keypoints from image 1 (N keypoints will exist).
            Corresponding keypoints from image 2 (there will also be N keypoints). These represent feature matches.
        """
        matcher = LoFTRKornia(pretrained=self._model_type)

        input = {"image0": self.to_tensor(image_i1), "image1": self.to_tensor(image_i2)}
        correspondences_dict = matcher(input)

        keypoints_i1 = Keypoints(coordinates=np.array(correspondences_dict[KEYPOINTS_I1_COORDINATES_KEY]))
        keypoints_i2 = Keypoints(coordinates=np.array(correspondences_dict[KEYPOINTS_I2_COORDINATES_KEY]))

        return keypoints_i1, keypoints_i2

    def to_tensor(self, image: Image) -> torch.Tensor:
        single_channel_value_array = image_utils.rgb_to_gray_cv(image).value_array.astype(np.float32) / 255.0
        return torch.from_numpy(single_channel_value_array)[None, None, :, :]
