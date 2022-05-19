"""Pose SLAM initialization module from relative pose priors.

Authors: Akshay Krishnan, Frank Dellaert
"""

from typing import Dict, List, Optional, Tuple

import dask
import gtsam
import numpy as np
from dask.delayed import Delayed
from gtsam import Pose3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.constraint import Constraint
from gtsfm.common.pose_prior import PosePrior, PosePriorType
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

PRIOR_NOISE_SIGMAS = [0.001, 0.001, 0.001, 0.1, 0.1, 0.1]
POSE_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_NOISE_SIGMAS)

logger = logger_utils.get_logger()


class PoseSlam:
    """Initializes using PoseSLAM given relative pose priors."""

    @staticmethod
    def angle(R1, R2):
        """Calculate angle between two rotations, in degrees."""
        return np.degrees(np.linalg.norm(R1.logmap(R2)))

    @staticmethod
    def difference(P1, P2):
        """Calculate the translation and angle differences of two poses.
        P1, P2: Pose3
        Return:
            distance: translation difference
            angle: angular difference
        """
        # TODO(frank): clean this up
        t1 = P1.translation()
        t2 = P2.translation()
        R1 = P1.rotation()
        R2 = P2.rotation()
        R1_2 = R1.compose(R2.inverse())
        t1_ = R1_2.rotate(t2)
        # t1_2 = t1 - R1_2*t2
        distance = np.linalg.norm(t1 - t1_)
        angle_ = PoseSlam.angle(R1, R2)
        return distance, angle_

    @staticmethod
    def check_covariance(cov: np.ndarray) -> bool:
        """Return false if covariance is bad."""
        if np.isnan(cov).any():
            return False
        try:
            info = np.linalg.inv(cov)
            if np.isnan(info).any():
                return False
            else:
                return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def filter_constraints(constraints: List[Constraint], poses: List[Pose3]) -> List[Constraint]:
        """Filter constraints according to notebook criteria.

        Args:
            constraints (List[Constraint]): relative constraints from file
            poses (List[Pose3]): estimated initial values

        Returns:
            Filtered constraints.
        """

        filtered_constraints: List[Constraint] = []

        # Filter on covariance and error
        for constraint in constraints:
            a, b = constraint.a, constraint.b
            aTb, cov = constraint.aTb, constraint.cov
            predicted_aTb = poses[a].between(poses[b])
            trans_diff, rot_diff = PoseSlam.difference(aTb, predicted_aTb)
            inlier = (trans_diff <= 0.04) and (rot_diff <= 5)
            if inlier and PoseSlam.check_covariance(cov):
                filtered_constraints.append(constraint)

        return filtered_constraints

    @staticmethod
    def filtered_pose_priors(
        constraints: List[Constraint], poses: List[Pose3], add_backbone=True
    ) -> Dict[Tuple[int, int], PosePrior]:
        """Generate relative pose priors from constraints and initial_estimate by filtering heavily.

        Args:
            constraints (List[Constraint]): relative constraints from file
            poses (List[Pose3]): estimated initial values
            add_backbone (bool, optional): Add soft backbone constraints. Defaults to True.

        Returns:
            Dict[Tuple[int, int], PosePrior]: dictionary of relative pose priors.
        """

        relative_pose_priors: Dict[Tuple[int, int], PosePrior] = {}
        for constraint in PoseSlam.filter_constraints(constraints, poses):
            a, b = constraint.a, constraint.b
            aTb, cov = constraint.aTb, constraint.cov
            relative_pose_priors[(a, b)] = PosePrior(aTb, cov, PosePriorType.SOFT_CONSTRAINT)

        # Create loose odometry factors for backbone.
        if add_backbone:
            backbone_cov = np.diag(np.array([np.deg2rad(30.0)] * 3 + [1] * 3))
            for i in range(len(poses) - 1):
                a, b = i, i + 1
                if (a, b) not in relative_pose_priors:
                    aTb = poses[a].between(poses[b])
                    relative_pose_priors[(a, b)] = PosePrior(aTb, backbone_cov, PosePriorType.SOFT_CONSTRAINT)

        # Output the resulting poses and pose constraints
        return relative_pose_priors

    def run_pose_slam(
        self,
        num_images: int,
        relative_pose_priors: Dict[Tuple[int, int], PosePrior] = {},
        gt_wTi_list: Optional[List[Optional[Pose3]]] = None,
    ) -> Tuple[List[Optional[Pose3]], Optional[GtsfmMetricsGroup]]:
        """Run the translation averaging.

        Args:
            num_images: number of camera poses.
            relative_pose_priors: priors on the pose between camera pairs as (i1, i2): i1Ti2.
            gt_wTi_list: List of ground truth poses (wTi) for computing metrics.

        Returns:
            List of Optional[Pose3],
            GtsfmMetricsGroup of 1DSfM metrics.
        """
        logger.info("[pose slam] Running pose SLAM intilialization")
        pose_init_graph = gtsam.NonlinearFactorGraph()

        for (i1, i2), i1Ti2_prior in relative_pose_priors.items():
            if self.check_covariance(i1Ti2_prior.covariance):
                pose_init_graph.push_back(
                    gtsam.BetweenFactorPose3(
                        i1,
                        i2,
                        i1Ti2_prior.value,
                        gtsam.noiseModel.Gaussian.Covariance(i1Ti2_prior.covariance),
                    )
                )
            else:
                logger.info(f"[pose slam] bad covariance between {i1} and {i2}")

        pose_init_graph.push_back(gtsam.PriorFactorPose3(0, Pose3(), POSE_PRIOR_NOISE))

        initial_values = gtsam.InitializePose3.initialize(pose_init_graph)
        initial_error = pose_init_graph.error(initial_values)
        logger.info(f"[pose slam] Pose SLAM initialization complete with error: {initial_error}")
        optimizer = gtsam.LevenbergMarquardtOptimizer(pose_init_graph, initial_values)
        result = optimizer.optimizeSafely()

        poses = [result.atPose3(i) if result.exists(i) else None for i in range(num_images)]
        final_error = pose_init_graph.error(result)
        logger.info(f"[pose slam] Pose SLAM optimization complete with error: {final_error}")

        return poses, GtsfmMetricsGroup(
            "pose_slam_metrics", [GtsfmMetric("intial_error", initial_error), GtsfmMetric("final_error", final_error)]
        )

    def create_computation_graph(
        self,
        num_images: int,
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        gt_wTi_list: Optional[List[Optional[Pose3]]] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing translation averaging.

        Args:
            num_images: number of camera poses.
            relative_pose_priors: priors on the pose between camera pairs (not delayed).
            gt_wTi_list: List of ground truth poses (wTi) for computing metrics.

        Returns:
            List of Optional[Pose3], wrapped as Delayed
            GtsfmMetricsGroup of 1DSfM metrics, wrapped as Delayed
        """
        return dask.delayed(self.run_pose_slam, nout=2)(num_images, relative_pose_priors, gt_wTi_list)
