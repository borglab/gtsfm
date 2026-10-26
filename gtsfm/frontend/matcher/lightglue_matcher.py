"""LightGlue matcher implementation

The network was proposed in 'LightGlue: Local Feature Matching at Light Speed' and is implemented by wrapping over
author's source-code.

Note: the pretrained model only supports SuperPoint or DISK detections currently.

References:
- https://github.com/cvg/LightGlue

Authors: Travis Driver
"""

from threading import Lock
from typing import Optional, Tuple

import numpy as np
import torch

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase


class LightGlueMatcher(MatcherBase):
    """Implements the LightGlue matcher -- a pretrained graph neural network using attention."""

    def __init__(self, features: str, use_cuda: bool = True):
        """Initialize the configuration and the parameters."""
        super().__init__()
        self._use_cuda = use_cuda
        self._features = features
        self._model: Optional[torch.nn.Module] = None  # Lazy loading - only create when needed
        self._model_lock = Lock()

    def __getstate__(self):
        """Exclude the process-local lock when sending this matcher to Dask workers."""
        state = self.__dict__.copy()
        state.pop("_model_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model_lock = Lock()

    def _ensure_model_loaded(self):
        """Lazy loading of the LightGlue model to avoid import warnings when using cache."""
        with self._model_lock:
            if self._model is None:
                from thirdparty.LightGlue.lightglue.lightglue import LightGlue

                logger = logger_utils.get_logger()
                logger.info("⏳ Loading LightGlue model weights...")
                device = torch.device("cuda" if self._use_cuda and torch.cuda.is_available() else "cpu")
                # Publish only a fully initialized model. Concurrent pair matches share it read-only;
                # moving or replacing it during inference can mix CPU and CUDA parameters.
                self._model = LightGlue(features=self._features).eval().to(device)

    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int, int],
        im_shape_i2: Tuple[int, int, int],
    ) -> np.ndarray:
        """Match keypoints using their 2D positions and descriptor vectors.

        Output format:
        1. Each row represents a match.
        2. The first column of row `k` represents the keypoint index from image #i1, for the k'th match.
        3. The second column of row `k` represents the keypoint index from image #i2, for the k'th match.
        4. Matches are sorted in descending order of the confidence (score), if possible.

        Args:
            keypoints_i1: Keypoints for image #i1, of length N1.
            keypoints_i2: Keypoints for image #i2, of length N2.
            descriptors_i1: Descriptors corresponding to keypoints_i1.
            descriptors_i2: Descriptors corresponding to keypoints_i2.
            im_shape_i1: Shape of image #i1, as (height, width).
            im_shape_i2: Shape of image #i2, as (height, width).

        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """
        # Ensure model is loaded only when actually needed
        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        device = next(self._model.parameters()).device

        if keypoints_i1.responses is None or keypoints_i2.responses is None:
            raise ValueError("Responses for keypoints required for LightGlue.")

        # Create dummy images.
        H1, W1, _ = im_shape_i1
        H2, W2, _ = im_shape_i2
        empty_image_i1 = torch.empty((1, 1, H1, W1))  # only used to determine dimensions for normalization
        empty_image_i2 = torch.empty((1, 1, H2, W2))

        # Build feature dictionaries.
        feats_i1 = {
            "keypoints": torch.from_numpy(keypoints_i1.coordinates).unsqueeze(0).float().to(device),
            "keypoint_scores": torch.from_numpy(keypoints_i1.responses).unsqueeze(0).float().to(device),
            "descriptors": torch.from_numpy(descriptors_i1).unsqueeze(0).float().to(device),
            "image": empty_image_i1.to(device),
        }
        feats_i2 = {
            "keypoints": torch.from_numpy(keypoints_i2.coordinates).unsqueeze(0).float().to(device),
            "keypoint_scores": torch.from_numpy(keypoints_i2.responses).unsqueeze(0).float().to(device),
            "descriptors": torch.from_numpy(descriptors_i2).unsqueeze(0).float().to(device),
            "image": empty_image_i2.to(device),
        }

        # Match!
        with torch.no_grad():
            assert self._model is not None, "Model should be loaded by now"
            matches = self._model({"image0": feats_i1, "image1": feats_i2})

        # Import rbd when needed to avoid triggering warnings on import
        from thirdparty.LightGlue.lightglue.utils import rbd

        feats_i1, feats_i2, matches = [rbd(x) for x in [feats_i1, feats_i2, matches]]  # remove batch dimension
        matches = matches["matches"].detach().cpu().numpy()  # indices with shape (N, 2)

        return matches
