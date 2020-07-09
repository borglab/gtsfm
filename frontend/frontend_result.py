"""
Class to hold the result (fundamental matrices and detected feature points)
between pairs of camera inputs.

This class contains an optional getter for relative poses (fundamental matrix ->
essential matrix -> relative pose) if camera instrinsics are provided.

Authors: Ayush Baid
"""
from typing import Dict, List, Tuple

import gtsam
import numpy as np

import frontend.utils.frontend_utils as frontend_utils


class FrontEndResult:
    """
    Class to hold F-matrix and matching feature points between pairs of camera
    poses.

    This class also exposes APIs based on these two inputs.
    """

    def __init__(self,
                 fundamental_matrices: Dict[Tuple[int, int], np.ndarray],
                 feature_points: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]):
        """
        Initializes the result.

        Args:
            fundamental_matrices (Dict[Tuple[int, int], np.ndarray]): 
                fundamental matrices between pairs of images with the tuple of image indices as the key.
            feature_points (Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]): 
                geometrically verified matching feature points between the two images. Note that the number of points
                for any pair of images should be the same from both the images.
        """
        self.fundamental_matrices = fundamental_matrices
        self.feature_points = feature_points

    def get_relative_poses(self,
                           intrinsics: List[np.ndarray]
                           ) -> Dict[Tuple[int, int], gtsam.Pose3]:
        """
        Compute relative poses between cameras using camera instrinsics.

        Args:
            intrinsics (List[np.ndarray]): calibration matrix for each matrix.

        Returns:
            Dict[Tuple[int, int], gtsam.Pose3]: relative pose between pairs of cameras.
        """

        poses = dict()

        for (idx1, idx2), f_matrix in self.fundamental_matrices.items():
            # compute the essential matrix from the fundamental matrix
            essential_mat = frontend_utils.essential_matrix_from_fundamental_matrix(
                f_matrix, intrinsics[idx1], intrinsics[idx2])

            # recover the pose
            points1, points2 = self.feature_points[(idx1, idx2)]
            poses[(idx1, idx2)] = frontend_utils.decompose_essential_matrix(
                essential_mat, points1, points2)

        return poses
