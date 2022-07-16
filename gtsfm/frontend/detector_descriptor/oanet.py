"""Wrapper around Order Aware Net verifier from Zhang et al.

See: https://arxiv.org/abs/1908.04964
Wraps the MIT-licensed code from https://github.com/zjhthu/OANet.

Authors: Ayush Baid, John Lambert
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
from gtsam import Cal3Bundler, Rot3, Unit3

import gtsfm.frontend.verifier.verifier_base as verifier_base
import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from thirdparty.OANet.core.oan import OANet

logger = logger_utils.get_logger()

OANET_MODEL_FPATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "thirdparty"
    / "OANet"
    / "weights"
    / "gl3d"
    / "sift-4000"
    / "model_best.pth"
)


class OANetVerifier(VerifierBase):
    def __init__(
        self,
        use_intrinsics_in_verification: bool,
        estimation_threshold_px: float,
    ) -> None:
        """Initializes the verifier.

        Due to our implementation structure, providing the descriptors to the verifier is not possible
        (thus ratio test/mutual info unavailable).

        Args:
            use_intrinsics_in_verification: Flag to perform keypoint normalization and compute the essential matrix
                instead of fundamental matrix. This should be preferred when the exact intrinsics are known as opposed
                to approximating them from exif data.
            estimation_threshold_px: maximum distance (in pixels) to consider a match an inlier, under squared
                Sampson distance.
        """
        self._use_intrinsics_in_verification = use_intrinsics_in_verification
        self._estimation_threshold_px = estimation_threshold_px
        self._min_matches = (
            verifier_base.NUM_MATCHES_REQ_E_MATRIX
            if self._use_intrinsics_in_verification
            else verifier_base.NUM_MATCHES_REQ_F_MATRIX
        )

        # for failure, i2Ri1 = None, and i2Ui1 = None, and no verified correspondences, and inlier_ratio_est_model = 0
        self._failure_result = (None, None, np.array([], dtype=np.uint64), 0.0)

        self.is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda else "cpu")

        self.model_path = OANET_MODEL_FPATH

        self.default_config = {}
        self.default_config["net_channels"] = 128
        self.default_config["net_depth"] = 12
        self.default_config["clusters"] = 500
        self.default_config["use_ratio"] = 0
        self.default_config["use_mutual"] = 0  # TODO: add option to use ratio and mutual as side info.
        self.default_config["iter_num"] = 1
        self.default_config["inlier_threshold"] = 1
        self.default_config_ = SimpleNamespace(**self.default_config)

        self.model = OANet(self.default_config_, self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])

        if self.is_cuda:
            self.model = self.model.cuda()

        self.model.eval()

        # Run RANSAC after OANet, for additional verification.
        self.ransac = Ransac(
            use_intrinsics_in_verification=use_intrinsics_in_verification,
            estimation_threshold_px=estimation_threshold_px,
        )

    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, float]:
        """Performs verification of correspondences between two images to recover the relative pose and indices of
        verified correspondences.

        Based on OANet's LearnedMatcher class: https://github.com/zjhthu/OANet/blob/master/demo/learnedmatcher.py

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3. These are subset of match_indices.
            Inlier ratio w.r.t. the estimated model, i.e. #ransac inliers / # putative matches.
        """
        if match_indices.shape[0] < self._min_matches:
            logger.info("[OANet] Not enough correspondences for verification.")
            return self._failure_result

        if len(keypoints_i1) < self._min_matches:
            return self._failure_result

        # only matched keypoints are fed into the model.
        matched_keypoints_i1 = keypoints_i1.coordinates[match_indices[:, 0]]
        matched_keypoints_i2 = keypoints_i2.coordinates[match_indices[:, 1]]

        # using OANet's custom normalization utility, obtain two (N,2) arrays
        normalized_keypoints = [
            torch.from_numpy(self.__normalize_kpts(matched_keypoints_i1).astype(np.float32)).to(self.device),
            torch.from_numpy(self.__normalize_kpts(matched_keypoints_i2).astype(np.float32)).to(self.device),
        ]

        # (N,4) -> (1,1,N,4)
        corr = torch.cat(normalized_keypoints, dim=-1)
        corr = corr.unsqueeze(0).unsqueeze(0)

        data = {}
        data["xs"] = corr
        data["sides"] = []  # using no side-information.

        with torch.no_grad():
            # obtain length-2 list back `y_hat`, each a Tensor of shape (1,N)
            y_hat, _ = self.model(data)

        y = y_hat[-1][0, :].cpu().numpy()
        inlier_mask = y > self.default_config_.inlier_threshold

        v_corr_idxs = match_indices[inlier_mask]
        inlier_ratio_est_model = np.mean(inlier_mask)

        if len(v_corr_idxs) < 8:
            return self._failure_result

        # Call F or E matrix verifier on OANet output.
        i2Ri1, i2Ui1, _, _ = self.ransac.verify(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            match_indices=v_corr_idxs,
            camera_intrinsics_i1=camera_intrinsics_i1,
            camera_intrinsics_i2=camera_intrinsics_i2,
        )
        return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio_est_model

    def __normalize_kpts(self, kpts: np.ndarray) -> np.ndarray:
        """Normalize keypoint coordinates to have zero-mean and a scale of sqrt(2).

        Args:
            kpts: array of shape (N,2) representing **unnormalized** 2d keypoint coordinates.

        Returns:
            nkpts: array of shape (N,2) representing **normalized** 2d keypoint coordinates.
        """
        x_mean = np.mean(kpts, axis=0)
        dist = kpts - x_mean
        meandist = np.linalg.norm(dist, axis=1).mean()
        scale = np.sqrt(2) / meandist
        T = np.zeros([3, 3])
        T[0, 0], T[1, 1], T[2, 2] = scale, scale, 1
        T[0, 2], T[1, 2] = -scale * x_mean[0], -scale * x_mean[1]
        nkpts = kpts * np.asarray([T[0, 0], T[1, 1]]) + np.array([T[0, 2], T[1, 2]])

        return nkpts
