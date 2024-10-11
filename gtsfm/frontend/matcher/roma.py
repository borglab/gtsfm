"""RoMa image matcher.

The network was proposed in "RoMa: Revisiting Robust Losses for Dense Feature Matching".

References:
- https://arxiv.org/html/2305.15404v2

Authors: Travis Driver
"""
from typing import Tuple

import numpy as np
import PIL
import torch
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase
from romatch import roma_indoor, roma_outdoor


class RoMa(ImageMatcherBase):
    """RoMa image matcher."""

    def __init__(
        self,
        use_cuda: bool = True,
        min_confidence: float = 0.1,
        max_keypoints: int = 8000,
        use_indoor_model: bool = False,
    ) -> None:
        """Initialize the matcher.

        Args:
            use_outdoor_model (optional): use the outdoor pretrained model. Defaults to True.
            use_cuda (optional): use CUDA for inference on GPU. Defaults to True.
            min_confidence(optional): Minimum confidence required for matches. Defaults to 0.95.
            upsample_res: resolution of upsampled warp and certainty maps. Stored as (H, W).
        """
        super().__init__()
        self._min_confidence = min_confidence
        self._max_keypoints = max_keypoints

        # Initialize model.
        self._device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        if use_indoor_model:
            self._matcher = roma_indoor(self._device).eval()
        else:
            self._matcher = roma_outdoor(self._device).eval()

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
        # Compute dense warp and certainty maps.
        with torch.no_grad():
            im1 = PIL.Image.fromarray(image_i1.value_array).convert("RGB")
            im2 = PIL.Image.fromarray(image_i2.value_array).convert("RGB")
            warp, certainty = self._matcher.match(im1, im2, device=self._device)

        # Sample keypoints and correspondences from warp.
        H1, W1 = image_i1.shape[:2]
        H2, W2 = image_i2.shape[:2]
        match, certs = self._matcher.sample(warp, certainty, num=self._max_keypoints)
        match = match[certs > self._min_confidence]
        mkpts1, mkpts2 = self._matcher.to_pixel_coordinates(match, H1, W1, H2, W2)

        # Convert to GTSfM keypoints and filter by mask.
        keypoints_i1 = Keypoints(coordinates=mkpts1.cpu().numpy())
        keypoints_i2 = Keypoints(coordinates=mkpts2.cpu().numpy())
        valid_ind = np.arange(len(keypoints_i1))
        if image_i1.mask is not None:
            _, valid_ind_i1 = keypoints_i1.filter_by_mask(image_i1.mask)
            valid_ind = np.intersect1d(valid_ind, valid_ind_i1)
        if image_i2.mask is not None:
            _, valid_ind_i2 = keypoints_i2.filter_by_mask(image_i2.mask)
            valid_ind = np.intersect1d(valid_ind, valid_ind_i2)

        return keypoints_i1.extract_indices(valid_ind), keypoints_i2.extract_indices(valid_ind)
