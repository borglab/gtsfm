"""
Order-Aware Net verifier implementation.

The detector was proposed in 'Learning Two-View Correspondences and Geometry Using Order-Aware Network' and is implemented by wrapping over the author's implementation

References:
- https://arxiv.org/abs/1908.04964 
- https://github.com/zjhthu/OANet

Authors: Ayush Baid
"""

import os
from collections import namedtuple
from typing import Tuple

import cv2 as cv
import numpy as np
import torch

from frontend.verifier.base_verifier import VerifierBase
from thirdparty.implementation.oanet.core.oan import OANet


class OANetVerifier(VerifierBase):
    """OA-Net verifier."""

    def __init__(self, is_cuda=True):
        super().__init__(min_pts=8)

        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda else "cpu")

        self.model_path = os.path.abspath(os.path.join(
            'thirdparty', 'models', 'oanet', 'gl3d', 'sift-4000', 'model_best.pth'))

        self.default_config = {}
        self.default_config['net_channels'] = 128
        self.default_config['net_depth'] = 12
        self.default_config['clusters'] = 500
        self.default_config['use_ratio'] = 0  # not using ratio
        self.default_config['use_mutual'] = 0  # not using mutual
        self.default_config['iter_num'] = 1
        self.default_config['inlier_threshold'] = 1
        self.default_config_ = namedtuple("Config", self.default_config.keys())(
            *self.default_config.values())

        self.model = OANet(self.default_config_, self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

        if self.is_cuda:
            self.model = self.model.cuda()

        self.model.eval()

    def verify(self,
               matched_features_im1: np.ndarray,
               matched_features_im2: np.ndarray,
               image_shape_im1: Tuple[int, int],
               image_shape_im2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the geometric verification of the matched features.

        Args:
            matched_features_im1 (np.ndarray): matched features from image #1
            matched_features_im2 (np.ndarray): matched features from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2

        Returns:
            np.ndarray: estimated fundamental matrix
            np.ndarray: index of the match features which are verified
        """
        fundamental_matrix = None
        verified_indices = np.array([], dtype=np.uint32)

        try:
            if matched_features_im1.shape[0] < self.min_pts:
                return fundamental_matrix, verified_indices

            features_im1_ = matched_features_im1[:, :2]
            features_im2_ = matched_features_im2[:, :2]

            with torch.no_grad():
                normalized_keypoints = [
                    torch.from_numpy(self.__normalize_kpts(
                        features_im1_).astype(np.float32)).to(self.device),
                    torch.from_numpy(self.__normalize_kpts(
                        features_im2_).astype(np.float32)).to(self.device)
                ]

                corr = torch.cat(normalized_keypoints, dim=-1)

                corr = corr.unsqueeze(0).unsqueeze(0)

                data = {}
                data['xs'] = corr
                data['sides'] = []

                y_hat, _ = self.model(data)
                y = y_hat[-1][0, :].cpu().numpy()
                verified_indices = np.where(
                    y > self.default_config['inlier_threshold'])[0].astype(np.uint32)

                # TODO: confirm with John if this is correct
                if verified_indices.size >= self.min_pts:
                    inlier_pts1 = features_im1_[verified_indices]
                    inlier_pts2 = features_im2_[verified_indices]

                    # We have the points as well as the essential matrix estimate; we have to recover the fundamental matrix now
                    fundamental_matrix, _ = cv.findFundamentalMat(
                        inlier_pts1, inlier_pts2, method=cv.FM_8POINT
                    )
                else:
                    verified_indices = np.array([], dtype=np.uint32)

        except Exception as e:
            print(e)

        return fundamental_matrix, verified_indices

    def __normalize_kpts(self, kpts):
        x_mean = np.mean(kpts, axis=0)
        dist = kpts - x_mean
        meandist = np.sqrt((dist**2).sum(axis=1)).mean()
        scale = np.sqrt(2) / meandist
        T = np.zeros([3, 3])
        T[0, 0], T[1, 1], T[2, 2] = scale, scale, 1
        T[0, 2], T[1, 2] = -scale*x_mean[0], -scale*x_mean[1]
        nkpts = kpts * np.asarray([T[0, 0], T[1, 1]]) + \
            np.array([T[0, 2], T[1, 2]])
        return nkpts
