"""
Class which holds the results for the front-end.

Authors: Ayush Baid
"""
from typing import Dict, Tuple

import cv2 as cv
import gtsam
import numpy as np

from loader.loader_base import LoaderBase


class FrontEndResult:
    def __init__(self,
                 loader: LoaderBase,
                 fundamental_matrices: Dict[Tuple[int, int], np.ndarray],
                 feature_points: Dict[Tuple[int, int],
                                      Tuple[np.ndarray, np.ndarray]]
                 ):
        self.loader = loader

        self.fundamental_matrices = fundamental_matrices
        self.feature_points = feature_points

        self.relative_poses = self.__relative_pose_from_fundamental_matrix()

    def __relative_pose_from_fundamental_matrix(self) -> Dict[Tuple[int, int], gtsam.Pose3]:

        poses = dict()

        for (idx1, idx2), f_matrix in self.fundamental_matrices.items():
            # compute the essential matrix from the fundamental matrix
            e_matrix = self.loader.get_instrinsics(
                idx2).T @ f_matrix @ self.loader.get_instrinsics(idx1)

            # recover the pose using opencv
            points1, points2 = self.feature_points[(idx1, idx2)]
            _, R, t, _ = cv.recoverPose(
                e_matrix, points1[:, :2], points2[:, :2])

            # store the poses in a dictionary
            poses[(idx1, idx2)] = gtsam.Pose3(
                gtsam.Rot3(R), gtsam.Point3(t.flatten())
            )

        return poses

    def get_relative_rotations(self) -> Dict[Tuple[int, int], gtsam.Rot3]:
        return {k: v.rotation() for k, v in self.relative_poses.items()}

    def get_relative_translations(self) -> Dict[Tuple[int, int], gtsam.Point3]:
        return {k: v.translation() for k, v in self.relative_poses.items()}
