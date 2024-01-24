"""
Uncertainity aware rotation averaging.

This algorithm was proposed in "Revisiting Rotation Averaging: Uncertainties and Robust Losses" and is implemented by 
following the authors' source code.

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
from gtsfm.common.pose_prior import PosePrior

import gtsfm.utils.logger as logger_utils
import gtsfm.averaging.rotation.const as rotation_const
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase

logger = logger_utils.get_logger()

DUMMY_COVARIANCE = np.eye(3)


def random_rotation() -> Rot3:
    """Sample a random rotation by generating a sample from the 4d unit sphere."""
    q = np.random.randn(4) * 0.03
    # make unit-length quaternion
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    R = Rot3(qw, qx, qy, qz)
    return R


class UncertainityAwareRotationAveraging(RotationAveragingBase):
    def run_rotation_averaging(
        self,
        num_images: int,
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]],
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior],
    ) -> List[Optional[Rot3]]:
        rotation_info_dict: Dict[Tuple[int, int], RotationInfo] = {
            k: RotationInfo(i2Ri1=v, covariance_mat=DUMMY_COVARIANCE) for k, v in i2Ri1_dict.items() if v is not None
        }

        global_rotations_init = [Rot3()] + [random_rotation() for _ in range(num_images - 1)]
        # global_rotations_init = [
        #     Rot3.RzRyRx(0, 0, 0) * random_rotation(),
        #     Rot3.RzRyRx(0, np.deg2rad(30), 0) * random_rotation(),
        #     i2Ri1_dict[(1, 0)].compose(i2Ri1_dict[(2, 1)]) * random_rotation(),
        # ]

        return _estimate_rotations_with_customized_loss_and_covariance_ceres(global_rotations_init, rotation_info_dict)


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


class MAGSACWeightBasedLoss(pyceres.LossFunction):
    def __init__(self, sigma, inverse=False):
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
        # rho = np.zeros((3, 1), dtype=np.float64)
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


def _estimate_rotations_with_customized_loss_and_covariance_ceres(
    global_orientations_init: List[Rot3],
    two_view_rotations: Dict[Tuple[int, int], RotationInfo],
    rotation_error_type: RotationErrorType = RotationErrorType.ANGLE_AXIS_COVARIANCE,
) -> List[Rot3]:
    if len(two_view_rotations) == 0:
        logger.warning("Skipping nonlinear rotation optimization because no relative rotations were provided.")

    # Set up the problem and loss function.
    # ceres_options = pyceres.ProblemOptions()
    # ceres_options.
    problem = pyceres.Problem()

    if rotation_error_type != RotationErrorType.ANGLE_AXIS_COVARIANCE:
        raise NotImplementedError(f"Rotation error type {rotation_error_type} not implemented")

    optimization_result = [(wRi.inverse().toQuaternion().coeffs(), np.zeros(3)) for wRi in global_orientations_init]

    for view_id_pair, rotation_info in two_view_rotations.items():
        i1, i2 = view_id_pair
        rotation1 = global_orientations_init[i1]
        rotation2 = global_orientations_init[i2]
        covariance = rotation_info.covariance_mat

        # Do not add the relative rotation constraint if it requires an orientation
        # that we do not have an initialization for.
        if (
            rotation1 is None
            or rotation2 is None
            or (covariance is None and rotation_error_type == RotationErrorType.ANGLE_AXIS_COVARIANCE)
            or (covariance is None and rotation_error_type == RotationErrorType.ANGLE_AXIS_COV_INLIERS)
            or (covariance is None and rotation_error_type == RotationErrorType.ANGLE_AXIS_COVTRACE)
            or (covariance is None and rotation_error_type == RotationErrorType.ANGLE_AXIS_COVNORM)
        ):
            continue

        costs = []
        if rotation_error_type == RotationErrorType.ANGLE_AXIS_COVARIANCE:
            pose_cov = np.eye(6)
            pose_cov[:3, :3] = covariance
            cost = pyceres.factors.PoseGraphRelativeCost(
                np.array(rotation_info.i2Ri1.toQuaternion().coeffs()), np.zeros(3), pose_cov
            )
            costs.append(cost)
            problem.add_residual_block(
                cost,
                pyceres.TrivialLoss(),
                # MAGSACWeightBasedLoss(sigma=0.02, inverse=False),
                [
                    optimization_result[i1][0],
                    optimization_result[i1][1],
                    optimization_result[i2][0],
                    optimization_result[i2][1],
                ],
            )

        else:
            logger.warning("Rotation error type %s skipped", rotation_error_type)

    problem.set_parameter_block_constant(optimization_result[0][0])
    problem.set_parameter_block_constant(optimization_result[0][1])
    for i in range(len(global_orientations_init)):
        problem.set_manifold(optimization_result[i][0], pyceres.QuaternionManifold())
    # problem.set_parameter_block_constant(optimization_result[i][1])
    # problem.set_parameter_block_constant(optimization_result[0][1])

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = True
    options.num_threads = -1
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    logger.info(summary.BriefReport())

    print("Optimization result at end", optimization_result)

    wRi = [Rot3(quat[0], quat[1], quat[2], quat[3]).inverse() for quat, _ in optimization_result]

    # return global_orientations_init

    return wRi
