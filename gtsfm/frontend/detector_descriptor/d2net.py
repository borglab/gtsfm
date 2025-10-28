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
PYRAMID_SCALES = [0.5, 1, 2]

MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "thirdparty" / "d2net" / "weights" / "d2_tf.pth"


class D2NetDetDesc(DetectorDescriptorBase):
    """D2-Net detector descriptor."""

    def __init__(self, max_keypoints: int = 5000, model_path: Path = MODEL_PATH, use_cuda: bool = True) -> None:
        """Instantiate parameters and hardware settings for D2-Net detector-descriptor.

        We set the maximum number of keypoints, set the path to pre-trained weights, and determine whether
        CUDA enabled devices can be utilized for inference.
        """
        super().__init__()
        self.max_keypoints = max_keypoints
        self.model_path = model_path
        self.use_cuda = use_cuda

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"D2-Net weights not found at {model_path}. "
                f"Please run 'bash scripts/download_model_weights.sh' from the repo root."
            )

        self._model = D2Net(model_file=self.model_path, use_relu=USE_RELU, use_cuda=False).eval()

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
        device = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")
        self._model.to(device)

        # Resize image, and obtain re-scaling factors to postprocess keypoint coordinates.
        resized_image, fact_i, fact_j = resize_image(image.value_array)
        input_image = preprocess_image(resized_image, preprocessing=PREPROCESSING_METHOD)

        scales = PYRAMID_SCALES if USE_MULTISCALE else [1]

        with torch.no_grad():
            keypoints, scores, descriptors = d2net_pyramid.process_multiscale(
                image=torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32)).to(device),
                model=self._model,
                scales=scales,
            )

        # Choose the top K keypoints and descriptors.
        ordered_idxs = np.argsort(-scores)[: self.max_keypoints]
        keypoints = keypoints[ordered_idxs, :]
        descriptors = descriptors[ordered_idxs, :]
        scores = scores[ordered_idxs]

        # Rescale keypoint coordinates from resized image scale, back to provided input image resolution.
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j

        # Convert (y,x) tuples that represented (i, j) indices of image matrix, into (u, v) coordinates.
        keypoints = keypoints[:, [1, 0]]
        return Keypoints(coordinates=keypoints, responses=scores), descriptors


def resize_image(
    image: np.ndarray, max_edge_px: int = MAX_EDGE_PX, max_sum_edges_px: int = MAX_SUM_EDGES_PX
) -> Tuple[np.ndarray, float, float]:
    """Resize image to limit to a maximum edge length and maximum summed length of both edges.

    Args:
        image: array of shape (H,W,3) or (H,W) representing an RGB or grayscale image.
        max_edge_px: maximum allowed edge length (in pixels).
        max_sum_edges_px: maximum allowed summed length of both edges (in pixels).

    Returns:
        resized_image: resized image array of shape (H1,W1,3).
        fact_i: vertical re-scaling factor for keypoint x-coordinates (inverse of downsampling factor).
        fact_j: horizontal re-scaling factor for keypoint y-coordinates.
    """
    if len(image.shape) == 2:
        # If grayscale, repeat grayscale channel 3 times.
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    resized_image = image
    # Downsample if maximum edge length or sum of edges exceeds specified thresholds.
    if max(resized_image.shape) > max_edge_px:
        resized_image = scipy.misc.imresize(resized_image, max_edge_px / max(resized_image.shape)).astype("float")
    if sum(resized_image.shape[:2]) > max_sum_edges_px:
        resized_image = scipy.misc.imresize(resized_image, max_sum_edges_px / sum(resized_image.shape[:2])).astype(
            "float"
        )

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]
    return resized_image, fact_i, fact_j
