"""Pose SLAM initialization module from relative pose priors.

Authors: Akshay Krishnan, Frank Dellaert
"""
from typing import Dict, List, Optional, Tuple

import dask
import gtsam
from dask.delayed import Delayed
from gtsam import Pose3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.pose_prior import PosePrior
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

PRIOR_NOISE_SIGMAS = [0.001, 0.001, 0.001, 0.1, 0.1, 0.1]
POSE_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_NOISE_SIGMAS)

logger = logger_utils.get_logger()


class PoseSlam:
    def __init__(self, use_gt_for_initialization: bool = False) -> None:
        self._use_gt_for_initialization = use_gt_for_initialization

    def __initialize_poses(
        self,
        pose_init_graph: gtsam.NonlinearFactorGraph,
        gt_wTi_list: Optional[List[Optional[Pose3]]] = None,
    ) -> gtsam.Values:
        if self._use_gt_for_initialization and gt_wTi_list is not None:
            logger.info("Using GT for initialization")
            initial_values = gtsam.Values()
            for i, wTi in enumerate(gt_wTi_list):
                if wTi is None:
                    continue

                initial_values.insert(i, wTi)

            return initial_values

        # TODO: handle case of partial GT information.
        initial_values = gtsam.InitializePose3.initialize(pose_init_graph, initial_values, useGradient=False)

        return initial_values

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
            pose_init_graph.push_back(
                gtsam.BetweenFactorPose3(
                    i1,
                    i2,
                    i1Ti2_prior.value,
                    gtsam.noiseModel.Gaussian.Covariance(i1Ti2_prior.covariance),
                )
            )
        pose_init_graph.push_back(gtsam.PriorFactorPose3(0, Pose3(), POSE_PRIOR_NOISE))
        initial_values = self.__initialize_poses(pose_init_graph, gt_wTi_list)
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
