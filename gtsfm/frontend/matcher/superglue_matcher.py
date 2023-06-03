"""SuperGlue matcher implementation

The network was proposed in 'SuperGlue: Learning Feature Matching with Graph Neural Networks' and is implemented by 
wrapping over author's source-code.

Note: the pretrained model only supports SuperPoint detections currently.

References:
- http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf # noqa
- https://github.com/magicleap/SuperGluePretrainedNetwork

Authors: Ayush Baid, John Lambert
"""
from typing import Tuple

import numpy as np
import torch

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from thirdparty.SuperGluePretrainedNetwork.models.superglue import SuperGlue

SUPERGLUE_DESC_DIM = 256
# hyperparameter below set per the author's default demo recommendations
DEFAULT_NUM_SINKHORN_ITERATIONS = 20


class SuperGlueMatcher(MatcherBase):
    """Implements the SuperGlue matcher -- a pretrained graph neural network using attention."""

    def __init__(self, use_cuda: bool = True, use_outdoor_model: bool = True):
        """Initialize the configuration and the parameters."""
        super().__init__()

        self._config = {
            "descriptor_dim": SUPERGLUE_DESC_DIM,
            "weights": "outdoor" if use_outdoor_model else "indoor",
            "sinkhorn_iterations": DEFAULT_NUM_SINKHORN_ITERATIONS,
        }
        self._use_cuda = use_cuda and torch.cuda.is_available()
        self._model = SuperGlue(self._config).eval()

    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> np.ndarray:
        """Match keypoints using their 2d positions and descriptor vectors.

        Output format:
        1. Each row represents a match.
        2. The first column of row `k` represents the keypoint index from image #i1, for the k'th match.
        3. The second column of row `k` represents the keypoint index from image #i2, for the k'th match.
        4. Matches are sorted in descending order of the confidence (score), if possible.

        Args:
            keypoints_i1: keypoints for image #i1, of length N1.
            keypoints_i2: keypoints for image #i2, of length N2.
            descriptors_i1: descriptors corr. to keypoints_i1.
            descriptors_i2: descriptors corr. to keypoints_i2.
            im_shape_i1: shape of image #i1, as (height,width).
            im_shape_i2: shape of image #i2, as (height,width).

        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """
        if keypoints_i1.responses is None or keypoints_i2.responses is None:
            raise ValueError("Responses for keypoints required for SuperGlue")

        if descriptors_i1.shape[1] != SUPERGLUE_DESC_DIM or descriptors_i2.shape[1] != SUPERGLUE_DESC_DIM:
            raise Exception("Superglue pretrained network only works on 256 dimensional descriptors")

        # batch size and number of channels
        B, C = 1, 1

        H1, W1 = im_shape_i1
        H2, W2 = im_shape_i2
        # feed in dummy arguments, as they are only needed to determine image dimensions for normalization
        empty_image_i1 = torch.empty((B, C, H1, W1))
        empty_image_i2 = torch.empty((B, C, H2, W2))

        input_data = {
            "keypoints0": torch.from_numpy(keypoints_i1.coordinates).unsqueeze(0).float(),
            "keypoints1": torch.from_numpy(keypoints_i2.coordinates).unsqueeze(0).float(),
            "descriptors0": torch.from_numpy(descriptors_i1).T.unsqueeze(0).float(),
            "descriptors1": torch.from_numpy(descriptors_i2).T.unsqueeze(0).float(),
            "scores0": torch.from_numpy(keypoints_i1.responses).unsqueeze(0).float(),
            "scores1": torch.from_numpy(keypoints_i2.responses).unsqueeze(0).float(),
            "image0": empty_image_i1,
            "image1": empty_image_i2,
        }

        with torch.no_grad():
            pred = self._model(input_data)
            matches = pred["matches0"][0].detach().cpu().numpy()

            num_kps_i1 = len(keypoints_i1)
            num_kps_i2 = len(keypoints_i2)
            valid = matches > -1
            match_indices = np.hstack(
                [np.arange(num_kps_i1)[valid].reshape(-1, 1), np.arange(num_kps_i2)[matches[valid]].reshape(-1, 1)]
            ).astype(np.uint32)

        return match_indices
