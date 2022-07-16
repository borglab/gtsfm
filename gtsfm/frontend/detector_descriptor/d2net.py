"""Implements the D2Net detector-descriptor by wrapping around the original Pytorch source code.

Paper: https://arxiv.org/abs/1905.03561
https://github.com/mihaidusmanu/d2-net

Authors: John Lambert
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

import thirdparty.d2net.lib.pyramid as d2net_pyramid
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase
from thirdparty.d2net.lib.model_test import D2Net
from thirdparty.d2net.lib.utils import preprocess_image

MAX_EDGE_PX = 1600
MAX_SUM_EDGES_PX = 2800
USE_MULTISCALE = False
USE_RELU = True
PREPROCESSING_METHOD = "torch"

MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "thirdparty" / "d2net" / "weights" / "d2_tf.pth"


class D2NetDetDesc(DetectorDescriptorBase):
    """D2-Net detector descriptor."""

    def __init__(self, max_keypoints: int = 5000, model_path: Path = MODEL_PATH, use_cuda: bool = True) -> None:
        """ """
        super().__init__()
        self.max_keypoints = max_keypoints
        self.model_path = model_path
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Extract keypoints and their corresponding descriptors.

        Adapted from:
        https://github.com/mihaidusmanu/d2-net/blob/master/extract_features.py

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """
        model = D2Net(model_file=self.model_path, use_relu=USE_RELU, use_cuda=self.use_cuda)
        model.eval()

        resized_image, fact_i, fact_j = resize_image(image.value_array)
        input_image = preprocess_image(resized_image, preprocessing=PREPROCESSING_METHOD)
        with torch.no_grad():
            if USE_MULTISCALE:
                keypoints, scores, descriptors = d2net_pyramid.process_multiscale(
                    torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32), device=self.device), model
                )
            else:
                keypoints, scores, descriptors = d2net_pyramid.process_multiscale(
                    torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32), device=self.device),
                    model,
                    scales=[1],
                )

        # choose the top K keypoints.
        ordered_idxs = np.argsort(-scores)[: self.max_keypoints]

        keypoints = keypoints[ordered_idxs, :]
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]
        keypoints = keypoints[:, :2]
        descriptors = descriptors[ordered_idxs, :]

        keypoints = Keypoints(coordinates=keypoints)
        return keypoints, descriptors


def resize_image(image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Resize image to limit to a maximum edge length and maximum summed length of both edges.

    Args:
        image: array of shape (H,W,3) representing an RGB image.

    Returns:
        resized_image: resized image.
        fact_i: vertical scaling factor.
        fact_j: horizontal scaling factor.
    """
    if len(image.shape) == 2:
        # if grayscale, repeat grayscale channel 3 times.
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO(johnwlambert): Move it to a common utility
    resized_image = image
    if max(resized_image.shape) > MAX_EDGE_PX:
        resized_image = scipy.misc.imresize(resized_image, MAX_EDGE_PX / max(resized_image.shape)).astype("float")
    if sum(resized_image.shape[:2]) > MAX_SUM_EDGES_PX:
        resized_image = scipy.misc.imresize(resized_image, MAX_SUM_EDGES_PX / sum(resized_image.shape[:2])).astype(
            "float"
        )

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]
    return resized_image, fact_i, fact_j
