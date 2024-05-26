"""Shonan Rotation Averaging.

The algorithm was proposed in "Shonan Rotation Averaging:Global Optimality by
Surfing SO(p)^n" and is implemented by wrapping up over implementation provided
by GTSAM.

References:
- https://arxiv.org/abs/2008.02737
- https://gtsam.org/

Authors: Jing Wu, Ayush Baid, John Lambert
"""

from typing import Dict, List, Optional, Set, Tuple

import gtsam
import numpy as np
from gtsam import (
    LevenbergMarquardtParams,
    Rot3,
    ShonanAveraging3,
    ShonanAveragingParameters3,
    Values,
)

import gtsfm.utils.logger as logger_utils
import gtsfm.utils.rotation as rotation_util
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.common.pose_prior import PosePrior

ROT3_DOF = 3
POSE3_DOF = 6

logger = logger_utils.get_logger()

_DEFAULT_TWO_VIEW_ROTATION_SIGMA = 1.0


class ShonanRotationAveraging(RotationAveragingBase):
    """Performs Shonan rotation averaging."""

    def __init__(
        self,
        two_view_rotation_sigma: float = _DEFAULT_TWO_VIEW_ROTATION_SIGMA,
        weight_by_inliers: bool = True,
        use_mst_init: bool = True,
    ) -> None:
        """Initializes module.

        Note: `p_min` and `p_max` describe the minimum and maximum relaxation rank.

        Args:
            two_view_rotation_sigma: Covariance to use (lower values -> more strictly adhere to input measurements).
            weight_by_inliers: Whether to weight pairwise costs according to an uncertainty equal to the inverse number
                of inlier correspondences per edge.
        """
        super().__init__()
        self._p_min = 3
        self._p_max = 64
        self._two_view_rotation_sigma = two_view_rotation_sigma
        self._weight_by_inliers = weight_by_inliers
        self._use_mst_init = use_mst_init

    def __get_shonan_params(self) -> ShonanAveragingParameters3:
        lm_params = LevenbergMarquardtParams.CeresDefaults()
        shonan_params = ShonanAveragingParameters3(lm_params)
        shonan_params.setUseHuber(False)
        shonan_params.setCertifyOptimality(True)
        return shonan_params

    def __measurements_from_2view_relative_rotations(
        self,
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        num_correspondences_dict: Dict[Tuple[int, int], int],
    ) -> gtsam.BinaryMeasurementsRot3:
        """Create between factors from relative rotations computed by the 2-view estimator."""
        # TODO: how to weight the noise model on relative rotations compared to priors?

        # Default noise model if `self._weight_by_inliers` is False, or zero correspondences on edge.
        noise_model = gtsam.noiseModel.Isotropic.Sigma(ROT3_DOF, self._two_view_rotation_sigma)

        measurements = gtsam.BinaryMeasurementsRot3()
        for (i1, i2), i2Ri1 in i2Ri1_dict.items():
            if i2Ri1 is None:
                continue
            if self._weight_by_inliers and num_correspondences_dict[(i1, i2)] > 0:
                # ignore translation during rotation averaging
                noise_model = gtsam.noiseModel.Isotropic.Sigma(ROT3_DOF, 1 / num_correspondences_dict[(i1, i2)])

            measurements.append(gtsam.BinaryMeasurementRot3(i2, i1, i2Ri1, noise_model))

        return measurements

    def _measurements_from_pose_priors(
        self, i1Ti2_priors: Dict[Tuple[int, int], PosePrior], old_to_new_idxs: Dict[int, int]
    ) -> gtsam.BinaryMeasurementsRot3:
        """Create between factors from the priors on relative poses."""
        measurements = gtsam.BinaryMeasurementsRot3()

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
            noise_model = gtsam.noiseModel.Isotropic.Sigma(ROT3_DOF, noise_model_sigma)
            measurements.append(gtsam.BinaryMeasurementRot3(i1_, i2_, i1Ti2_prior.value.rotation(), noise_model))

        return measurements

    def _run_with_consecutive_ordering(
        self, num_connected_nodes: int, measurements: gtsam.BinaryMeasurementsRot3, initial: Optional[Values]
    ) -> List[Optional[Rot3]]:
        """Run the rotation averaging on a connected graph w/ N keys ordered consecutively [0,...,N-1].

        Note: GTSAM requires the N input nodes to be connected and ordered from [0 ... N-1].
        Modifying GTSAM would require a major philosophical overhaul, so we perform the re-ordering
        here in a sort of "wrapper". See https://github.com/borglab/gtsam/issues/784 for more details.

        Args:
            num_connected_nodes: Number of unique connected nodes (i.e. images) in the graph
                (<= the number of images in the dataset)
            measurements: BinaryMeasurementsRot3 created from relative rotations from 2-view estimator and the priors.

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
        shonan = ShonanAveraging3(measurements, self.__get_shonan_params())

        if initial is None:
            logger.info("Using random initialization for Shonan")
            initial = shonan.initializeRandomly()
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
        for i1, i2 in i2Ri1_dict.keys():
            unique_nodes_with_edges.add(i1)
            unique_nodes_with_edges.add(i2)
        for i1, i2 in relative_pose_priors.keys():
            unique_nodes_with_edges.add(i1)
            unique_nodes_with_edges.add(i2)

        return unique_nodes_with_edges

    def run_rotation_averaging(
        self,
        num_images: int,
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]],
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior],
        v_corr_idxs: Dict[Tuple[int, int], np.ndarray],
    ) -> List[Optional[Rot3]]:
        """Run the rotation averaging on a connected graph with arbitrary keys, where each key is a image/pose index.

        Note: functions as a wrapper that re-orders keys to prepare a graph w/ N keys ordered [0,...,N-1].
        All input nodes must belong to a single connected component, in order to obtain an absolute pose for each
        camera in a single, global coordinate frame.

        Args:
            num_images: Number of images. Since we have one pose per image, it is also the number of poses.
            i2Ri1_dict: Relative rotations for each image pair-edge as dictionary (i1, i2): i2Ri1.
            i1Ti2_priors: Priors on relative poses.
            v_corr_idxs: Dict mapping image pair indices (i1, i2) to indices of verified correspondences.

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
        old_to_new_idxs = {old_idx: i for i, old_idx in enumerate(nodes_with_edges)}

        i2Ri1_dict_remapped = {
            (old_to_new_idxs[i1], old_to_new_idxs[i2]): i2Ri1 for (i1, i2), i2Ri1 in i2Ri1_dict.items()
        }
        num_correspondences_dict: Dict[Tuple[int, int], int] = {
            (old_to_new_idxs[i1], old_to_new_idxs[i2]): len(v_corr_idxs[(i1, i2)])
            for (i1, i2) in v_corr_idxs.keys()
            if (i1, i2) in i2Ri1_dict
        }

        # Use negative of the number of correspondences as the edge weight.
        initial_values: Optional[Values] = None
        if self._use_mst_init:
            logger.info("Using MST initialization for Shonan")
            wRi_initial_ = rotation_util.initialize_global_rotations_using_mst(
                len(nodes_with_edges),
                i2Ri1_dict_remapped,
                edge_weights={
                    (i1, i2): -num_correspondences_dict.get((i1, i2), 0) for i1, i2 in i2Ri1_dict_remapped.keys()
                },
            )
            initial_values = Values()
            for i, wRi in enumerate(wRi_initial_):
                initial_values.insert(i, wRi)

        def _create_factors_and_run() -> List[Rot3]:
            measurements: gtsam.BinaryMeasurementsRot3 = self.__measurements_from_2view_relative_rotations(
                i2Ri1_dict=i2Ri1_dict_remapped, num_correspondences_dict=num_correspondences_dict
            )
            measurements.extend(self._measurements_from_pose_priors(i1Ti2_priors, old_to_new_idxs))
            wRi_list_subset = self._run_with_consecutive_ordering(
                num_connected_nodes=len(nodes_with_edges), measurements=measurements, initial=initial_values
            )
            return wRi_list_subset

        try:
            wRi_list_subset = _create_factors_and_run()
        except RuntimeError:
            logger.exception("Shonan failed")
            if self._weight_by_inliers is True:
                logger.info("Reattempting Shonan without inlier-weighted costs...")
                # At times, Shonan's `SparseMinimumEigenValue` fails to compute minimum eigenvalue.
                self._weight_by_inliers = False
                wRi_list_subset = _create_factors_and_run()
        wRi_list = [None] * num_images
        for remapped_i, original_i in enumerate(nodes_with_edges):
            wRi_list[original_i] = wRi_list_subset[remapped_i]

        return wRi_list
