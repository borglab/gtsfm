"""Shonan Rotation Averaging.

The algorithm was proposed in "Shonan Rotation Averaging:Global Optimality by
Surfing SO(p)^n" and is implemented by wrapping up over implementation provided
by GTSAM.

References:
- https://arxiv.org/abs/2008.02737
- https://gtsam.org/

Authors: Jing Wu, Ayush Baid, John Lambert
"""
from enum import Enum, unique
from typing import Dict, List, Optional, Set, Tuple, Any

import gtsam
import numpy as np
from gtsam import (
    BetweenFactorPose3,
    BetweenFactorPose3s,
    LevenbergMarquardtParams,
    Rot3,
    ShonanAveraging3,
    ShonanAveragingParameters3,
)

import gtsfm.utils.logger as logger_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.common.pose_prior import PosePrior

POSE3_DOF = 6
ROT3_DOF = 3

logger = logger_utils.get_logger()

_DEFAULT_TWO_VIEW_ROTATION_SIGMA = 1.0


@unique
class RobustErrorType(str, Enum):
    """Robust error function types for GTSAM.

    See https://gtsam.org/doxygen/4.0.0/a01426.html for more detailed information.
    """

    Huber: str = "Huber"
    GemanMcClure: str = "GemanMcClure"


# Stores default parameters for each error function.
DEFAULT_ROBUST_MODEL_KWARGS = {
    RobustErrorType.Huber: {"k": 0.1},
    RobustErrorType.GemanMcClure: {"c": 1.0},
}


class ShonanRotationAveraging(RotationAveragingBase):
    """Performs Shonan rotation averaging."""

    def __init__(
            self, 
            two_view_rotation_sigma: float = _DEFAULT_TWO_VIEW_ROTATION_SIGMA, 
            robust_error_model: Optional[RobustErrorType] = None,
            robust_error_model_kwargs: Optional[Dict[Any, Any]] = None,
        ) -> None:
        """Initializes module.

        Note: `p_min` and `p_max` describe the minimum and maximum relaxation rank.

        Args:
            two_view_rotation_sigma: Covariance to use (lower values -> more strictly adhere to input measurements).
        """
        self._two_view_rotation_sigma = two_view_rotation_sigma
        self._p_min = 5
        self._p_max = 64

        # Build robust error function.
        self._robust_error_model = robust_error_model
        if self._robust_error_model is not None:
            self._robust_error_model = getattr(gtsam.noiseModel.mEstimator, robust_error_model.value)
            if robust_error_model_kwargs is None:
                robust_error_model_kwargs = DEFAULT_ROBUST_MODEL_KWARGS[robust_error_model]
            self._robust_error_model = self._robust_error_model(**robust_error_model_kwargs)
            self._p_max = self._p_min


    def __get_shonan_params(self) -> ShonanAveragingParameters3:
        lm_params = LevenbergMarquardtParams.CeresDefaults()
        shonan_params = ShonanAveragingParameters3(lm_params)
        shonan_params.setUseHuber(self._robust_error_model is not None)
        shonan_params.setCertifyOptimality(self._robust_error_model is None)
        return shonan_params

    def __between_factors_from_2view_relative_rotations(
        self, 
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        corr_idxs: Dict[Tuple[int, int], np.ndarray],
        old_to_new_idxs: Dict[int, int],
        use_robust_error: bool,
    ) -> gtsam.BinaryMeasurementsRot3:
        """Create between factors from relative rotations computed by the 2-view estimator."""
        # TODO: how to weight the noise model on relative rotations compared to priors?
        measurements = gtsam.BinaryMeasurementsRot3()
        for (i1, i2), i2Ri1 in i2Ri1_dict.items():
            if i2Ri1 is not None and len(corr_idxs[(i1, i2)]) > 0:
                # ignore translation during rotation averaging
                noise_model = gtsam.noiseModel.Isotropic.Sigma(ROT3_DOF, 1 / corr_idxs[(i1, i2)].shape[0])
                if use_robust_error:
                    noise_model = gtsam.noiseModel.Robust(self._robust_error_model, noise_model)
                i2_ = old_to_new_idxs[i2]
                i1_ = old_to_new_idxs[i1]
                measurements.append(gtsam.BinaryMeasurementRot3(i2_, i1_, i2Ri1, noise_model))

        return measurements

    def _between_factors_from_pose_priors(
        self, i1Ti2_priors: Dict[Tuple[int, int], PosePrior], old_to_new_idxs: Dict[int, int]
    ) -> BetweenFactorPose3s:
        """Create between factors from the priors on relative poses."""
        between_factors = BetweenFactorPose3s()

        def get_isotropic_noise_model_sigma(covariance: np.ndarray) -> float:
            """Get the sigma to be used for the isotropic noise model.
            We compute the average of the diagonal entries of the covariance matrix.
            """
            avg_cov = np.average(np.diag(covariance), axis=None)
            return np.sqrt(avg_cov)

        for (i1, i2), i1Ti2_prior in i1Ti2_priors.items():
            i1_ = old_to_new_idxs[i1]
            i2_ = old_to_new_idxs[i2]
            noise_model_sigma = get_isotropic_noise_model_sigma(i1Ti2_prior.covariance)
            noise_model = gtsam.noiseModel.Isotropic.Sigma(POSE3_DOF, noise_model_sigma)
            between_factors.append(BetweenFactorPose3(i1_, i2_, i1Ti2_prior.value, noise_model))

        return between_factors

    def _run_with_consecutive_ordering(
        self, num_connected_nodes: int, 
        measurements: gtsam.BinaryMeasurementsRot3, 
        measurements_not_robust: gtsam.BinaryMeasurementsRot3,
    ) -> List[Optional[Rot3]]:
        """Run the rotation averaging on a connected graph w/ N keys ordered consecutively [0,...,N-1].

        Note: GTSAM requires the N input nodes to be connected and ordered from [0 ... N-1].
        Modifying GTSAM would require a major philosophical overhaul, so we perform the re-ordering
        here in a sort of "wrapper". See https://github.com/borglab/gtsam/issues/784 for more details.

        Args:
            num_connected_nodes: Number of unique connected nodes (i.e. images) in the graph
                (<= the number of images in the dataset)
            between_factors: BetweenFactorPose3s created from relative rotations from 2-view estimator and the priors.

        Returns:
            Global rotations for each **CONNECTED** camera pose, i.e. wRi, as a list. The number of entries in
                the list is `num_connected_nodes`. The list may contain `None` where the global rotation could
                not be computed (either underconstrained system or ill-constrained system).
        """

        logger.info(
            "Running Shonan with %d constraints on %d nodes",
            len(measurements),
            num_connected_nodes,
        )

        # Run without robust error.
        if self._robust_error_model is not None:
            lm_params = LevenbergMarquardtParams.CeresDefaults()
            shonan_params = ShonanAveragingParameters3(lm_params)
            shonan_params.setUseHuber(False)
            shonan_params.setCertifyOptimality(True)

            shonan = ShonanAveraging3(measurements_not_robust, shonan_params)
            initial = shonan.initializeRandomly()
            logger.info("Initial cost: %.5f", shonan.cost(initial))
            result, _ = shonan.run(initial, 5, 64)
            logger.info("Final cost: %.5f", shonan.cost(result))
            initial = result
        else:
            shonan = ShonanAveraging3(measurements, self.__get_shonan_params())
            initial = shonan.initializeRandomly()

        # Run with robust error.
        shonan = ShonanAveraging3(measurements, self.__get_shonan_params())
        logger.info("Initial cost: %.5f", shonan.cost(initial))
        result, _ = shonan.run(initial, self._p_min, self._p_max)
        logger.info("Final cost: %.5f", shonan.cost(result))

        wRi_list_consecutive = [None] * num_connected_nodes
        for i in range(num_connected_nodes):
            if result.exists(i):
                wRi_list_consecutive[i] = result.atRot3(i)

        return wRi_list_consecutive

    def _nodes_with_edges(
        self, i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]], relative_pose_priors: Dict[Tuple[int, int], PosePrior]
    ) -> Set[int]:
        """Gets the nodes with edges which are to be modelled as between factors."""

        unique_nodes_with_edges = set()
        for (i1, i2) in i2Ri1_dict.keys():
            unique_nodes_with_edges.add(i1)
            unique_nodes_with_edges.add(i2)
        for (i1, i2) in relative_pose_priors.keys():
            unique_nodes_with_edges.add(i1)
            unique_nodes_with_edges.add(i2)

        return unique_nodes_with_edges

    def run_rotation_averaging(
        self,
        num_images: int,
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]],
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior],
        corr_idxs: Dict[Tuple[int, int], np.ndarray],
    ) -> List[Optional[Rot3]]:
        """Run the rotation averaging on a connected graph with arbitrary keys, where each key is a image/pose index.

        Note: functions as a wrapper that re-orders keys to prepare a graph w/ N keys ordered [0,...,N-1].
        All input nodes must belong to a single connected component, in order to obtain an absolute pose for each
        camera in a single, global coordinate frame.

        Args:
            num_images: Number of images. Since we have one pose per image, it is also the number of poses.
            i2Ri1_dict: Relative rotations for each image pair-edge as dictionary (i1, i2): i2Ri1.
            i1Ti2_priors: Priors on relative poses.

        Returns:
            Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
                `num_images`. The list may contain `None` where the global rotation could not be computed (either
                underconstrained system or ill-constrained system), or where the camera pose had no valid observation
                in the input to run_rotation_averaging().
        """
        if len(i2Ri1_dict) == 0:
            logger.warning("Shonan cannot proceed: No cycle-consistent triplets found after filtering.")
            wRi_list = [None] * num_images
            return wRi_list

        nodes_with_edges = sorted(list(self._nodes_with_edges(i2Ri1_dict, i1Ti2_priors)))
        old_to_new_idxes = {old_idx: i for i, old_idx in enumerate(nodes_with_edges)}

        measurements: gtsam.BinaryMeasurementsRot3 = self.__between_factors_from_2view_relative_rotations(
            i2Ri1_dict, corr_idxs, old_to_new_idxes, self._robust_error_model is not None
        )
        measurements_not_robust: gtsam.BinaryMeasurementsRot3 = self.__between_factors_from_2view_relative_rotations(
            i2Ri1_dict, corr_idxs, old_to_new_idxes, False
        )
        # between_factors.extend(self._between_factors_from_pose_priors(i1Ti2_priors, old_to_new_idxes))

        wRi_list_subset = self._run_with_consecutive_ordering(
            num_connected_nodes=len(nodes_with_edges), 
            measurements=measurements, 
            measurements_not_robust=measurements_not_robust,
        )

        wRi_list = [None] * num_images
        for remapped_i, original_i in enumerate(nodes_with_edges):
            wRi_list[original_i] = wRi_list_subset[remapped_i]

        return wRi_list
