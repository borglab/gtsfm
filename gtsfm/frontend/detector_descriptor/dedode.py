"""Superpoint detector+descriptor implementation.

The network was proposed in 'SuperPoint: Self-Supervised Interest Point Detection and Description' and is implemented
by wrapping over the authors' implementation.

References:
- https://arxiv.org/abs/1712.07629
- https://github.com/magicleap/SuperGluePretrainedNetwork

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np
import torch
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import (
    DetectorDescriptorBase,
)
from kornia.feature import DeDoDe


class DeDoDeDetectorDescriptor(DetectorDescriptorBase):
    """Superpoint Detector+Descriptor implementation."""

    def __init__(self, max_keypoints: int = 10000, use_cuda: bool = True) -> None:
        """Configures the object.

        Args:
            max_keypoints: max keypoints to detect in an image.
            use_cuda (optional): flag controlling the use of GPUs via CUDA. Defaults to True.
            weights_path (optional): Path to the model weights. Defaults to MODEL_WEIGHT_PATH.
        """
        super().__init__(max_keypoints=max_keypoints)
        self._use_cuda = use_cuda
        self._model = DeDoDe.from_pretrained(
            detector_weights="L-upright", descriptor_weights="B-upright"
        ).eval()

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Jointly generate keypoint detections and their associated descriptors from a single image."""
        device = torch.device(
            "cuda" if self._use_cuda and torch.cuda.is_available() else "cpu"
        )
        self._model.to(device)

        # Compute features.
        image_tensor = (
            torch.from_numpy(image.value_array.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .float()
            .to(device)[None]
        )
        with torch.no_grad():
            coordinates, scores, descriptors = self._model(image_tensor)

        # Unpack results.
        coordinates = coordinates.squeeze(0).detach().cpu().numpy()
        scores = scores.squeeze(0).detach().cpu().numpy()
        keypoints = Keypoints(coordinates, scales=None, responses=scores)
        descriptors = descriptors.squeeze(0).detach().cpu().numpy()

        # Filter features.
        if image.mask is not None:
            keypoints, valid_idxs = keypoints.filter_by_mask(image.mask)
            descriptors = descriptors[valid_idxs]
        keypoints, selection_idxs = keypoints.get_top_k(self.max_keypoints)
        descriptors = descriptors[selection_idxs]

        return keypoints, descriptors
