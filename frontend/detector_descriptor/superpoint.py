"""
Superpoint detector+descriptor implementation

The network was proposed in 'SuperPoint: Self-Supervised Interest Point Detection and Description' and is implemented by wrapping over author's implementation.

References:
- https://arxiv.org/abs/1712.07629
- https://github.com/magicleap/SuperGluePretrainedNetwork

Authors: Ayush Baid
"""

from typing import Tuple

import numpy as np
import torch

import frontend.utils.image_utils as image_utils
from common.image import Image
from frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase
from thirdparty.implementation.superglue.models.superpoint import SuperPoint


class SuperPointImplementation(DetectorDescriptorBase):
    """Wrapper around the author's implementation."""

    def __init__(self, is_cuda=True):
        """Initialise the configuration and the parameters."""

        config = {
            'weights_path': 'thirdparty/models/superpoint/superpoint_v1.pth'
        }

        self.use_cuda = is_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = SuperPoint(config).to(self.device)

    def detect_and_describe(self, image: Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform feature detection as well as their description in a single step.

        Refer to detect() in BaseDetector and describe() in BaseDescriptor 
        for details about the output format.

        Args:
            image (Image): the input image

        Returns:
            Tuple[np.ndarray, np.ndarray]: detected features and their 
                                           descriptions as two numpy arrays
        """
        image_tensor = torch.from_numpy(
            np.expand_dims(image_utils.rgb_to_gray_matlab(image.image_array).astype(np.float32)/255.0, (0, 1))).to(self.device)
        model_results = self.model(image_tensor)

        features_points = model_results['keypoints'][0].detach().cpu().numpy()
        scores = model_results['scores'][0].detach().cpu().numpy()
        descriptors = model_results['descriptors'][0].detach().cpu().numpy().T

        detected_features = np.empty(
            (features_points.shape[0], 4), dtype=np.float32)
        detected_features[:, :2] = features_points
        detected_features[:, 3] = scores
        detected_features[:, 2] = np.NaN

        return detected_features, descriptors
