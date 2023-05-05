import logging
from typing import List, Optional

import gtsam
import numpy as np
from gtsam import (
    Rot3,
    Unit3,
    EssentialMatrixFactor,
    EssentialMatrix,
    NonlinearFactorGraph,
    PriorFactorPose3,
    Values,
    symbol_shorthand,
    CustomFactor,
)
import gtsfm.utils.features as feature_utils


X = symbol_shorthand.X  # camera pose

logger = logging.getLogger(__name__)


def refine_with_essential_constraint(
    keypoints_i1,
    keypoints_i2,
    camera_intrinsics_i1,
    camera_intrinsics_i2,
    i2Ri1: Rot3,
    i2Ui1: Unit3,
):
    """Refines the camera poses using essential matrix constraint.

    Args:
        i2Ri1: 3x3 rotation matrix from camera 1 to camera 2.
        i2Ui1: 3x1 translation vector from camera 1 to camera 2.
        i2Ei1: 3x3 essential matrix from camera 1 to camera 2.

    Returns:
        i2Ri1: 3x3 rotation matrix from camera 1 to camera 2.
        i2Ui1: 3x1 translation vector from camera 1 to camera 2.
    """
    uv_norm_i1 = feature_utils.normalize_coordinates(keypoints_i1.coordinates, camera_intrinsics_i1)
    uv_norm_i2 = feature_utils.normalize_coordinates(keypoints_i2.coordinates, camera_intrinsics_i2)

    initial = Values()
    initial.insert(X(1), gtsam.EssentialMatrix(i2Ri1, i2Ui1))

    graph = NonlinearFactorGraph()
    essential_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, 1e-3)
    for i in range(uv_norm_i1.shape[0]):
        graph.add(EssentialMatrixFactor(X(1), uv_norm_i2[i, :], uv_norm_i1[i, :], essential_noise_model))

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    i2Ei1 = result.atEssentialMatrix(X(1))
    i2Ri1 = i2Ei1.rotation()
    i2Ui1 = i2Ei1.direction()
    return i2Ri1, i2Ui1

    # graph.add(EssentialMatrixConstraint(X(2), X(1), i2Ei1, essential_noise_model))

    # # Add a prior at the origin, we only optimize relative pose.
    # graph.add(PriorFactorPose3(X(2), gtsam.Pose3(), gtsam.noiseModel.Isotropic.Sigma(6, 1e-8)))
    # # Relative translation is known only upto scale so add a prior on the magnitude of the translation.

    # def magnitude_error(this: CustomFactor, values: Values, jacobians: Optional[List[np.ndarray]]) -> float:
    #     """Error function for the prior on the relative translation magnitude."""
    #     pose1 = values.atPose3(this.keys()[0])
    #     t = pose1.translation()
    #     error = t[0] ** 2 + t[1] ** 2 + t[2] ** 2 - 1.0
    #     if jacobians is not None:
    #         jacobians[0] = np.array([0, 0, 0, 2 * t[0], 2 * t[1], 2 * t[2]])
    #     return error

    # mag_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, 1e-8)
    # mag_factor = CustomFactor(mag_noise_model, [X(1)], magnitude_error)
    # graph.add(mag_factor)

    # params = gtsam.LevenbergMarquardtParams()
    # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    # result = optimizer.optimize()
    # i2Ri1 = result.atPose3(X(1)).rotation()
    # i2Ui1 = result.atPose3(X(1)).translation()
    # return i2Ri1, i2Ui1
