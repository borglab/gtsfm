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
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase

KEYPOINTS_I1_COORDINATES_KEY = "keypoints0"
KEYPOINTS_I2_COORDINATES_KEY = "keypoints1"
CONFIDENCE_KEY = "confidence"


class LOFTR(ImageMatcherBase):
    """LOFTR image matcher."""

    def __init__(self, use_outdoor_model: bool = True, use_cuda: bool = True, min_confidence: float = 0.95) -> None:
        """Initialize the matcher.

        Args:
            use_outdoor_model (optional): use the outdoor pretrained model. Defaults to True.
            use_cuda (optional): use CUDA for inference on GPU. Defaults to True.
            min_confidence(optional): Minimum confidence required for matches. Defaults to 0.95.
        """
        super().__init__()
        self._model_type = "outdoor" if use_outdoor_model else "indoor"
        self._use_cuda: bool = use_cuda
        self._min_confidence = min_confidence
        self._matcher = LoFTRKornia(pretrained=self._model_type).eval()

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
        with torch.no_grad():
            input = {"image0": self.to_tensor(image_i1), "image1": self.to_tensor(image_i2)}
            correspondences_dict = self._matcher(input)

        coordinates_i1 = correspondences_dict[KEYPOINTS_I1_COORDINATES_KEY].cpu().numpy()
        coordinates_i2 = correspondences_dict[KEYPOINTS_I2_COORDINATES_KEY].cpu().numpy()
        match_confidence = correspondences_dict[CONFIDENCE_KEY].cpu().numpy()

        coords_i1_by_confidence, coords_i2_by_confidence = self.__sort_and_filter_by_confidence(
            coordinates_i1, coordinates_i2, match_confidence
        )

        keypoints_i1 = Keypoints(coordinates=coords_i1_by_confidence)
        keypoints_i2 = Keypoints(coordinates=coords_i2_by_confidence)
        valid_ind = np.arange(len(keypoints_i1))
        if image_i1.mask is not None:
            _, valid_ind_i1 = keypoints_i1.filter_by_mask(image_i1.mask)
            valid_ind = np.intersect1d(valid_ind, valid_ind_i1)
        if image_i2.mask is not None:
            _, valid_ind_i2 = keypoints_i2.filter_by_mask(image_i2.mask)
            valid_ind = np.intersect1d(valid_ind, valid_ind_i2)

        return keypoints_i1.extract_indices(valid_ind), keypoints_i2.extract_indices(valid_ind)

    def __sort_and_filter_by_confidence(
        self, coordinates_i1: np.ndarray, coordinates_i2: np.ndarray, match_confidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        idxs = np.argsort(-match_confidence)
        sorted_confs = match_confidence[idxs]

        num_vals_greater_than_threshold = (sorted_confs > self._min_confidence).sum()
        idxs = idxs[:num_vals_greater_than_threshold]

        return coordinates_i1[idxs], coordinates_i2[idxs]

    def to_tensor(self, image: Image) -> torch.Tensor:
        single_channel_value_array = image_utils.rgb_to_gray_cv(image).value_array.astype(np.float32) / 255.0
        return torch.from_numpy(single_channel_value_array)[None, None, :, :]
