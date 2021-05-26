"""Superpoint detector+descriptor implementation.

The network was proposed in 'SuperPoint: Self-Supervised Interest Point Detection and Description' and is implemented
by wrapping over author's implementation.

References:
- https://arxiv.org/abs/1712.07629
- https://github.com/magicleap/SuperGluePretrainedNetwork

Authors: Ayush Baid
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase
from nonfree.thirdparty.SuperGluePretrainedNetwork.models.superpoint import SuperPoint

ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent
MODEL_WEIGHTS_PATH = (
    ROOT_PATH / "nonfree" / "thirdparty" / "SuperGluePretrainedNetwork" / "models" / "weights" / "superpoint_v1.pth"
)


class SuperPointDetectorDescriptor(DetectorDescriptorBase):
    """Superpoint Detector+Descriptor implementation."""

    def __init__(self, use_cuda=True, weights_path=MODEL_WEIGHTS_PATH) -> None:
        """Configures the object.

        Args:
            max_keypoints: max keypoints to detect in an image.
            is_cuda (optional): flag controlling the use of GPUs via CUDA. Defaults to True.
            weights_path (optional): Path to the model weights. Defaults to MODEL_WEIGHT_PATH.
        """
        super().__init__()
        self._use_cuda = use_cuda and torch.cuda.is_available()
        self._config = {"weights_path": weights_path}

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        device = torch.device("cuda" if self._use_cuda else "cpu")
        model = SuperPoint(self._config).to(device)
        model.eval()
        # TODO: fix inference issue #110

        image_tensor = torch.from_numpy(
            np.expand_dims(image_utils.rgb_to_gray_cv(image).value_array.astype(np.float32) / 255.0, (0, 1))
        ).to(device)

        with torch.no_grad():
            model_results = model({"image": image_tensor})

        torch.cuda.empty_cache()

        features_points = model_results["keypoints"][0].detach().cpu().numpy()
        scores = model_results["scores"][0].detach().cpu().numpy()
        descriptors = model_results["descriptors"][0].detach().cpu().numpy().T

        # sort by scores
        sort_idx = np.argsort(-scores)
        features_points = features_points[sort_idx]
        scores = scores[sort_idx]
        descriptors = descriptors[sort_idx]

        # limit the number of keypoints
        features_points = features_points[: self.max_keypoints]
        scores = scores[: self.max_keypoints]
        descriptors = descriptors[: self.max_keypoints]

        keypoints = Keypoints(features_points, responses=scores, scales=np.ones(scores.shape))

        return keypoints, descriptors