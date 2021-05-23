
import numpy as np
import torch

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from thirdparty.SuperGluePretrainedNetwork.models.superglue import SuperGlue


class SuperGlueMatcher(MatcherBase):
    """Implements the SuperGlue matcher -- a pretrained graph neural network using attention."""

    def __init__(self, use_cuda: bool = False):
        super().__init__()
        self._use_cuda = use_cuda

        config = {
            "weights": "outdoor",
            "sinkhorn_iterations": 20,
        }
        self.superglue = SuperGlue(config).eval()

    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        img_height: int,
        img_width: int,
    ) -> np.ndarray:
        """Match keypoints using their 2d positions and descriptor vectors.

        Args:
            keypoints_i1: keypoints for image #i1, of length N1.
            keypoints_i2: keypoints for image #i2, of length N2.
            descriptors_i1: descriptors corr. to keypoints_i1.
            descriptors_i2: descriptors corr. to keypoints_i2.
            img_height: height of input images, in pixels. Assumes images i1 and i2 have the same size.
            img_width: width of input images, in pixels.

        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """
        device = torch.device("cuda" if self._use_cuda else "cpu")
        data = {}

        data["descriptors0"] = torch.from_numpy(descriptors_i1).T.unsqueeze(0).float().to(device)
        data["descriptors1"] = torch.from_numpy(descriptors_i2).T.unsqueeze(0).float().to(device)

        data["keypoints0"] = torch.from_numpy(keypoints_i1.coordinates).unsqueeze(0).float().to(device)
        data["keypoints1"] = torch.from_numpy(keypoints_i2.coordinates).unsqueeze(0).float().to(device)

        data["scores0"] = torch.from_numpy(keypoints_i1.responses).unsqueeze(0).float().to(device)
        data["scores1"] = torch.from_numpy(keypoints_i2.responses).unsqueeze(0).float().to(device)

        self.superglue = self.superglue.to(device)

        # batch size and number of channels
        B, C = 1, 1

        # feed in dummy arguments, as they are only needed to determine image dimensions for normalization
        data["image0"] = np.zeros((B, C, img_height, img_width))
        data["image1"] = np.zeros((B, C, img_height, img_width))

        with torch.no_grad():
            pred = self.superglue(data)

        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].cpu().numpy()

        num_kps_i1 = len(keypoints_i1)
        num_kps_i2 = len(keypoints_i2)
        valid = matches > -1
        match_indices = np.hstack(
            [np.arange(num_kps_i1)[valid].reshape(-1, 1), np.arange(num_kps_i2)[matches[valid]].reshape(-1, 1)]
        )
        return match_indices
