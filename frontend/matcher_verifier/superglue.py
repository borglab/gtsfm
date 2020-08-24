"""
SuperGlue matcher+verifier implementation

The network was proposed in 'SuperGlue: Learning Feature Matching with Graph Neural Networks' and is implemented by wrapping over author's implementation.

References:
- http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
- https://github.com/magicleap/SuperGluePretrainedNetwork

Authors: Ayush Baid
"""

from typing import Tuple

import cv2 as cv
import numpy as np
import pydegensac
import torch

from frontend.matcher_verifier.matcher_verifier_base import MatcherVerifierBase
from thirdparty.implementation.superglue.models.superglue import SuperGlue


class SuperGlueImplementation(MatcherVerifierBase):
    """Wrapper around the author's implementation."""

    # TODO: handle variable descriptor dimensions
    def __init__(self, is_cuda=True):
        """Initialise the configuration and the parameters."""

        config = {
            'descriptor_dim': 256,
            'weights_path': 'thirdparty/models/superglue/superglue_outdoor.pth'
        }

        self.use_cuda = is_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = SuperGlue(config).to(self.device)

    def match_and_verify(self,
                         features_im1: np.ndarray,
                         features_im2: np.ndarray,
                         descriptors_im1: np.ndarray,
                         descriptors_im2: np.ndarray,
                         image_shape_im1: Tuple[int, int],
                         image_shape_im2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Matches the features (using their corresponding descriptors) to return geometrically verified outlier-free correspondences as indices of input features.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corresponding descriptors from image #1
            descriptors_im2 (np.ndarray): corresponding descriptors from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2

        Returns:
            np.ndarray: estimated fundamental matrix
            np.ndarray: verified correspondences as index of the input features in a Nx2 array
        """

        return self.__compute(features_im1, features_im2, descriptors_im1, descriptors_im2, image_shape_im1, image_shape_im2)[:2]

    def match_and_verify_and_get_features(self, features_im1, features_im2, descriptors_im1, descriptors_im2, image_shape_im1, image_shape_im2):
        F, _, verified_features_im1, verified_features_im2 = self.__compute(
            features_im1, features_im2, descriptors_im1, descriptors_im2, image_shape_im1, image_shape_im2
        )

        return F, verified_features_im1, verified_features_im2

    def __compute(self,
                  features_im1: np.ndarray,
                  features_im2: np.ndarray,
                  descriptors_im1: np.ndarray,
                  descriptors_im2: np.ndarray,
                  image_shape_im1: Tuple[int, int],
                  image_shape_im2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Common function which performs computations for both the public APIs.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corresponding descriptors from image #1
            descriptors_im2 (np.ndarray): corresponding descriptors from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2

        Returns:
            np.ndarray: estimated fundamental matrix
            np.ndarray: verified correspondences as index of the input features in a Nx2 array
            np.ndarray: features from image #1 corresponding to verified match indices
            np.ndarray: features from image #2 corresponding to verified match indices

        """

        if features_im1.size == 0 or features_im2.size == 0:
            return None, np.array([], dtype=np.uint32)

        if features_im1.shape[1] < 4 or features_im2.shape[1] < 4:
            # we do not have the feature confidence as input
            raise Exception("No confidence score on detected features")

        if descriptors_im1.shape[1] != 256 or descriptors_im2.shape[1] != 256:
            print(descriptors_im1.shape)
            raise Exception(
                "Superglue pretrained network only works on 256 dimensional descriptors"
            )

        # convert to datatypes required by the forward function
        data = {
            'keypoints0': torch.from_numpy(np.expand_dims(features_im1[:, :2], 0)).to(self.device),
            'keypoints1': torch.from_numpy(np.expand_dims(features_im2[:, :2], 0)).to(self.device),
            'descriptors0': torch.from_numpy(np.expand_dims(np.transpose(descriptors_im1), 0)).to(self.device),
            'descriptors1': torch.from_numpy(np.expand_dims(np.transpose(descriptors_im2), 0)).to(self.device),
            'scores0': torch.from_numpy(np.expand_dims(features_im1[:, 3], (0))).to(self.device),
            'scores1': torch.from_numpy(np.expand_dims(features_im2[:, 3], (0))).to(self.device),
            'image_shape1': image_shape_im1,
            'image_shape2': image_shape_im2
        }

        superglue_results = self.model(data)

        matches_for_features_im1 = np.squeeze(
            superglue_results['matches0'].detach().cpu().numpy())

        match_indices_im1 = np.where(matches_for_features_im1 > -1)[0]
        match_indices_im2 = matches_for_features_im1[match_indices_im1]

        verified_indices = np.concatenate(
            [match_indices_im1.reshape(-1, 1), match_indices_im2.reshape(-1, 1)], axis=1).astype(np.uint32)

        verified_features_im1 = features_im1[verified_indices[:, 0], :2]
        verified_features_im2 = features_im2[verified_indices[:, 1], :2]

        # compute the Fundamental matrix using 8-point algorithm
        if(verified_features_im1.shape[0] < 8):
            fundamental_matrix = None
        else:
            fundamental_matrix, _ = cv.findFundamentalMat(
                verified_features_im1, verified_features_im2, method=cv.FM_8POINT)
            # print('MV input points #: {}, {}'.format(
            #     features_im1.shape[0], features_im2.shape[0]))
            # print('Ransac input points #: ', verified_features_im1.shape[0])
            # fundamental_matrix, mask = pydegensac.findFundamentalMatrix(
            #     verified_features_im1,
            #     verified_features_im2,
            #     1.0,
            #     0.999,
            #     enable_degeneracy_check=True
            # )

            # ransac_inlier_idx = mask.ravel() == 1
            # verified_indices = verified_indices[ransac_inlier_idx]
            # verified_features_im1 = verified_features_im1[ransac_inlier_idx]
            # verified_features_im2 = verified_features_im2[ransac_inlier_idx]

            # print('Ransac output points #: ', verified_features_im1.shape[0])

        return fundamental_matrix, verified_indices, verified_features_im1, verified_features_im2
