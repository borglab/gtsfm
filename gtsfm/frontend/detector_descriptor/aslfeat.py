"""Implements the ASLFeat detector-descriptor by wrapping around the original Pytorch source code.

Paper: https://arxiv.org/abs/1905.03561
https://github.com/mihaidusmanu/d2-net

Authors: Travis Driver
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import sys

import numpy as np
import tensorflow.compat.v1 as tf

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase

sys.path.append("thirdparty/ASLFeat")
from thirdparty.ASLFeat.models.feat_model import FeatModel


DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "thirdparty"
    / "ASLFeat"
    / "weights"
    / "aslfeatv2"
    / "model.ckpt-60000"
)
DEFAULT_CONFIG = {
    "net": {
        "max_dim": 2048,
        "config": {
            "kpt_n": 8000,
            "kpt_refinement": True,
            "deform_desc": 1,
            "score_thld": 0.5,
            "edge_thld": 10,
            "multi_scale": True,
            "multi_level": True,
            "nms_size": 3,
            "eof_mask": 5,
            "need_norm": True,
            "use_peakiness": True,
        },
    }
}


class ASLFeatDetectorDescriptor(DetectorDescriptorBase):
    """ASLFeat detector-descriptor."""

    def __init__(
        self,
        max_keypoints: int = 5000,
        model_path: str = DEFAULT_MODEL_PATH,
        model_config: Dict[str, Any] = DEFAULT_CONFIG,
    ) -> None:
        """Instantiate parameters and hardware settings for ASLFeat detector-descriptor."""
        super().__init__()
        self.max_keypoints = max_keypoints
        self.model_path = model_path
        self.model_config = model_config

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Extract keypoints and their corresponding descriptors.

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """
        tf.reset_default_graph()

        # Load model and image.
        model = FeatModel(self.model_path, **self.model_config["net"])
        image_tensor = image_utils.rgb_to_gray_cv(image).value_array[..., None]

        # Compute features.
        descriptors, keypoints, scores = model.run_test_data(image_tensor)
        keypoints = Keypoints(coordinates=keypoints, responses=scores.squeeze())

        # Filter features.
        if image.mask is not None:
            keypoints, valid_idxs = keypoints.filter_by_mask(image.mask)
            descriptors = descriptors[valid_idxs]
        keypoints, selection_idxs = keypoints.get_top_k(self.max_keypoints)
        descriptors = descriptors[selection_idxs]

        return keypoints, descriptors
