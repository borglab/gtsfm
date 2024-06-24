"""
Uncertainty aware rotation averaging.

This algorithm was proposed in "Revisiting Rotation Averaging: Uncertainties and Robust Losses" and is implemented by 
following the authors' source code (GNU GPL v3.0 license)

References:
- https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Revisiting_Rotation_Averaging_Uncertainties_and_Robust_Losses_CVPR_2023_paper.pdf
- https://github.com/zhangganlin/GlobalSfMpy
"""

import math
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pyceres
from gtsam import Rot3
import networkx as nx
from gtsfm.common.pose_prior import PosePrior

import gtsfm.utils.logger as logger_utils
import gtsfm.averaging.rotation.gamma_values as rotation_const
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase

logger = logger_utils.get_logger()

DUMMY_COVARIANCE = np.eye(3) * 1e-2


class RotationInfo(NamedTuple):
    i2Ri1: Rot3
    covariance_mat: np.ndarray


class RotationErrorType(Enum):
    QUATERNION_NORM = 0
    ROTATION_MAT_FNORM = 1
    QUATERNION_COSINE = 2
    ANGLE_AXIS_COVARIANCE = 3
    ANGLE_AXIS = 4
    ANGLE_AXIS_INLIERS = 5
    ANGLE_AXIS_COV_INLIERS = 6
    ANGLE_AXIS_COVTRACE = 7
    ANGLE_AXIS_COVNORM = 8


def random_rotation() -> Rot3:
    """Sample a random rotation by generating a sample from the 4d unit sphere."""
    q = np.random.randn(4) * 0.03
    # Make unit-norm quaternion.
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    R = Rot3(qw, qx, qy, qz)
    return R


def initialize_global_rotations_using_mst(num_images: int, i2Ri1_dict: Dict[Tuple[int, int], Rot3]) -> List[Rot3]:
    # Create a graph from the relative rotations dictionary
    graph = nx.Graph()
    for i1, i2 in i2Ri1_dict.keys():
        # TODO: use inlier count as weight
        graph.add_edge(i1, i2, weight=1)

    # Compute the Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(graph)

    wRis = [random_rotation() for _ in range(num_images)]
    for i1, i2 in sorted(mst.edges):
        if (i1, i2) in i2Ri1_dict:
            wRis[i2] = wRis[i1] * i2Ri1_dict[(i1, i2)].inverse()
        else:
            wRis[i2] = wRis[i1] * i2Ri1_dict[(i2, i1)]

    return wRis


class UncertaintyAwareRotationAveraging(RotationAveragingBase):
    def run_rotation_averaging(
        self,
        num_images: int,
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]],
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior],
        v_corr_idxs: Dict[Tuple[int, int], np.ndarray],
    ) -> List[Optional[Rot3]]:
        rotation_info_dict: Dict[Tuple[int, int], RotationInfo] = {
            k: RotationInfo(i2Ri1=v, covariance_mat=DUMMY_COVARIANCE) for k, v in i2Ri1_dict.items() if v is not None
        }

        global_rotations_init = initialize_global_rotations_using_mst(
            num_images, {key: value for key, value in i2Ri1_dict.items() if value is not None}
        )

        return _estimate_rotations_with_customized_loss_and_covariance_ceres(global_rotations_init, rotation_info_dict)


class MAGSACWeightBasedLoss(pyceres.LossFunction):
    """MAGSAC++ inspired loss proposed in "Revisiting Rotation Averaging: Uncertainties and Robust Losses"."""

    def __init__(self, sigma: float, inverse: bool = False) -> None:
        super().__init__()
        self.sigma_max = sigma
        self.nu = rotation_const.NU3
        self.squared_sigma = self.sigma_max * self.sigma_max
        self.squared_sigma_max_2 = 2.0 * self.squared_sigma
        self.cubed_sigma_max = self.squared_sigma * self.sigma_max
        self.dof_minus_one_per_two = (self.nu - 1.0) / 2.0
        self.C_times_two_ad_dof = rotation_const.C3 * (2**self.dof_minus_one_per_two)
        self.one_over_sigma = self.C_times_two_ad_dof / self.sigma_max
        self.gamma_value = math.gamma(self.dof_minus_one_per_two)
        self.gamma_difference = self.gamma_value - rotation_const.UPPER_INCOMPLETE_GAMMA_OF_K3
        self.weight_zero = self.one_over_sigma * self.gamma_difference
        self.use_weight_inverse = inverse

    def Evaluate(self, squared_residual, rho):
        zero_derivative = False
        if squared_residual > rotation_const.SIGMA_QUANTILE3 * rotation_const.SIGMA_QUANTILE3 * self.squared_sigma:
            squared_residual = rotation_const.SIGMA_QUANTILE3 * rotation_const.SIGMA_QUANTILE3 * self.squared_sigma
            zero_derivative = True

        x = round(rotation_const.PRECISION_OF_STORED_GAMMA3 * squared_residual / self.squared_sigma_max_2)
        if rotation_const.STORED_GAMMA_NUMBER3 < x:
            x = rotation_const.STORED_GAMMA_NUMBER3
        s = x * self.squared_sigma_max_2 / rotation_const.PRECISION_OF_STORED_GAMMA3

        weight = self.one_over_sigma * (
            rotation_const.STORED_GAMMA_VALUES3[x] - rotation_const.UPPER_INCOMPLETE_GAMMA_OF_K3
        )
        weight_derivative = (
            -self.C_times_two_ad_dof
            * ((s / self.squared_sigma_max_2) ** (self.nu / 2 - 1.5))
            * math.exp(-s / self.squared_sigma_max_2)
            / (2 * self.cubed_sigma_max)
        )
        if s < 1e-7:
            s = 1e-7
        weight_second_derivative = (
            2.0
            * self.C_times_two_ad_dof
            * ((s / self.squared_sigma_max_2) ** (self.nu / 2 - 1.5))
            * (1.0 / self.squared_sigma - (self.nu - 3) / s)
            * math.exp(-s / self.squared_sigma_max_2)
            / (8 * self.cubed_sigma_max)
        )

        if self.use_weight_inverse:
            rho[0] = 1.0 / weight
            rho[1] = -1.0 / (weight * weight) * weight_derivative
            rho[2] = 2.0 / (
                weight * weight * weight
            ) * weight_derivative * weight_derivative - weight_second_derivative / (weight * weight)
            if zero_derivative:
                rho[1] = 0.00001
                rho[2] = 0.0
        else:
            rho[0] = self.weight_zero - weight
            rho[1] = -weight_derivative
            rho[2] = -weight_second_derivative
            if rho[1] == 0:
                rho[1] = 0.00001
            if zero_derivative:
                rho[1] = 0.00001
                rho[2] = 0.0

        return rho


def _to_ceres_quat(rotation: Rot3) -> np.ndarray:
    gtsam_quat = rotation.toQuaternion()
    return np.array([gtsam_quat.w(), gtsam_quat.x(), gtsam_quat.y(), gtsam_quat.z()], dtype=np.float64)


def _to_gtsam_rot(ceres_quat: np.ndarray) -> Rot3:
    w, x, y, z = ceres_quat
    return Rot3(w, x, y, z)


def _estimate_rotations_with_customized_loss_and_covariance_ceres(
    global_orientations_init: List[Rot3],
    two_view_rotations: Dict[Tuple[int, int], RotationInfo],
    rotation_error_type: RotationErrorType = RotationErrorType.ANGLE_AXIS_COVARIANCE,
) -> List[Optional[Rot3]]:
    if len(two_view_rotations) == 0:
        logger.warning("Skipping nonlinear rotation optimization because no relative rotations were provided.")

    problem = pyceres.Problem()

    if rotation_error_type != RotationErrorType.ANGLE_AXIS_COVARIANCE:
        raise NotImplementedError(f"Rotation error type {rotation_error_type} not implemented")

    # Note(Ayush): In camera frame
    optimization_result = [(_to_ceres_quat(wRi.inverse()), np.zeros(3)) for wRi in global_orientations_init]

    valid_rotations = set()

    for view_id_pair, rotation_info in two_view_rotations.items():
        i1, i2 = view_id_pair
        valid_rotations.add(i1)
        valid_rotations.add(i2)
        rotation1 = global_orientations_init[i1]
        rotation2 = global_orientations_init[i2]
        covariance = rotation_info.covariance_mat

        # Do not add the relative rotation constraint if it requires an orientation
        # that we do not have an initialization for.
        if rotation1 is None or rotation2 is None:
            continue

        costs = []
        if rotation_error_type == RotationErrorType.ANGLE_AXIS_COVARIANCE:
            pose_cov = np.eye(6)
            pose_cov[:3, :3] = covariance
            cost = pyceres.factors.PoseGraphRelativeCost(_to_ceres_quat(rotation_info.i2Ri1), np.zeros(3), pose_cov)
            costs.append(cost)
            problem.add_residual_block(
                cost,
                pyceres.SoftLOneLoss(0.1),
                [
                    optimization_result[i1][0],
                    optimization_result[i1][1],
                    optimization_result[i2][0],
                    optimization_result[i2][1],
                ],
            )

        else:
            logger.warning("Rotation error type %s skipped", rotation_error_type)

    valid_rotations_list = list(valid_rotations)

    problem.set_parameter_block_constant(optimization_result[valid_rotations_list[0]][0])
    problem.set_parameter_block_constant(optimization_result[valid_rotations_list[0]][1])
    for i in valid_rotations_list:
        problem.set_manifold(optimization_result[i][0], pyceres.QuaternionManifold())

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = False
    options.num_threads = -1
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    logger.info(summary.BriefReport())

    wRis_list = [_to_gtsam_rot(ceres_quat).inverse() for ceres_quat, _ in optimization_result]

    return [wRi if i in valid_rotations else None for i, wRi in enumerate(wRis_list)]
