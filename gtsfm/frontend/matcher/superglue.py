"""SuperGlue matcher implementation

The network was proposed in 'SuperGlue: Learning Feature Matching with Graph Neural Networks' and is implemented by 
wrapping over author's source-code.

Note: the pretrained model only supports superpoint detections right now.

References:
- http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
- https://github.com/magicleap/SuperGluePretrainedNetwork

Authors: Ayush Baid
"""
import os
from typing import Tuple

import numpy as np
import torch

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from nonfree.thirdparty.SuperGluePretrainedNetwork.models.superglue import SuperGlue


class SuperGlueMatcher(MatcherBase):
    """SuperGlue matcher+verifier implementation."""

    def __init__(
        self, is_cuda=True, weights_path=os.path.join("thirdparty", "models", "superglue", "superglue_outdoor.pth"),
    ):
        """Initialise the configuration and the parameters."""
        super().__init__()

        self._config = {"descriptor_dim": 256, "weights_path": weights_path}

        self._use_cuda = is_cuda and torch.cuda.is_available()

    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> np.ndarray:
        """Match descriptor vectors.

        Output format:
        1. Each row represents a match.
        2. First column represents keypoint index from image #i1.
        3. Second column represents keypoint index from image #i2.
        4. Matches are sorted in descending order of the confidence (score), if possible.

        Args:
            keypoints_i1: keypoints for image #i1, of length N1.
            keypoints_i2: keypoints for image #i2, of length N2.
            descriptors_i1: descriptors corr. to keypoints_i1.
            descriptors_i2: descriptors corr. to keypoints_i2.
            im_shape_i1: shape of image #i1, as width, height.
            im_shape_i2: shape of image #i2, as width, height.

        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """

        if keypoints_i1.responses is None or keypoints_i2.responses is None:
            raise ValueError("Responses for keypoints required for SuperGlue")

        if descriptors_i1.shape[1] != 256 or descriptors_i2.shape[1] != 256:
            raise Exception("Superglue pretrained network only works on 256 dimensional descriptors")

        _device = torch.device("cuda" if self._use_cuda else "cpu")
        _model = SuperGlue(self._config).to(_device)
        _model.eval()

        with torch.no_grad():
            empty_image_i1 = torch.empty((1, 1, im_shape_i1[0], im_shape_i1[1]))
            empty_image_i2 = torch.empty((1, 1, im_shape_i2[0], im_shape_i2[1]))
            input_data = {
                "keypoints0": torch.from_numpy(np.expand_dims(keypoints_i1.coordinates, 0)).to(_device),
                "keypoints1": torch.from_numpy(np.expand_dims(keypoints_i2.coordinates, 0)).to(_device),
                "descriptors0": torch.from_numpy(np.expand_dims(np.transpose(descriptors_i1), 0)).to(_device),
                "descriptors1": torch.from_numpy(np.expand_dims(np.transpose(descriptors_i2), 0)).to(_device),
                "scores0": torch.from_numpy(np.expand_dims(keypoints_i1.responses, (0))).to(_device),
                "scores1": torch.from_numpy(np.expand_dims(keypoints_i2.responses, (0))).to(_device),
                "image0": empty_image_i1,
                "image1": empty_image_i2,
            }
            output_data = _model(input_data)
            indices0 = output_data["matches0"]
            matches_for_features_im1 = np.squeeze(indices0.detach().cpu().numpy())

            match_indices_im1 = np.where(matches_for_features_im1 > -1)[0]
            match_indices_im2 = matches_for_features_im1[match_indices_im1]

            verified_indices = np.concatenate(
                [match_indices_im1.reshape(-1, 1), match_indices_im2.reshape(-1, 1)], axis=1
            ).astype(np.uint32)

        torch.cuda.empty_cache()

        return verified_indices
