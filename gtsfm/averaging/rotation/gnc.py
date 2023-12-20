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
import scipy
from gtsam import (
    BetweenFactorPose3,
    BetweenFactorPose3s,
    Pose3,
    Rot3,
    ShonanAveraging3,
    GncLMOptimizer,
    GncLMParams
)

import gtsfm.utils.logger as logger_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.common.pose_prior import PosePrior

ROT3_DOF = 3
POSE3_DOF = 6

logger = logger_utils.get_logger()

_DEFAULT_TWO_VIEW_ROTATION_SIGMA = 1.0


class GncRotationAveraging(RotationAveragingBase):
    """Performs Shonan rotation averaging."""

    def __init__(self, two_view_rotation_sigma: float = _DEFAULT_TWO_VIEW_ROTATION_SIGMA) -> None:
        """Initializes module.

        Note: `p_min` and `p_max` describe the minimum and maximum relaxation rank.

        Args:
            two_view_rotation_sigma: Covariance to use (lower values -> more strictly adhere to input measurements).
        """
        self._two_view_rotation_sigma = two_view_rotation_sigma

    def __get_gnc_params(self) -> GncLMParams:
        params = GncLMParams()
        return params

    def __get_shonan_params(self) -> gtsam.ShonanAveragingParameters3:
        lm_params = gtsam.LevenbergMarquardtParams.CeresDefaults()
        shonan_params = gtsam.ShonanAveragingParameters3(lm_params)
        shonan_params.setUseHuber(False)
        shonan_params.setCertifyOptimality(True)
        return shonan_params

    def __graph_from_2view_relative_rotations(
        self, i2Ri1_dict: Dict[Tuple[int, int], Rot3], old_to_new_idxs: Dict[int, int]
    ) -> BetweenFactorPose3s:
        """Create between factors from relative rotations computed by the 2-view estimator."""
        # TODO: how to weight the noise model on relative rotations compared to priors?
        noise_model = gtsam.noiseModel.Isotropic.Sigma(ROT3_DOF, self._two_view_rotation_sigma)
        between_factors = gtsam.NonlinearFactorGraph()
        # graph.addPriorRot3(gtsam.symbol("R", 0), gtsam.Rot3(np.eye(3)), sigma_R0)

        for (i1, i2), i2Ri1 in i2Ri1_dict.items():
            if i2Ri1 is not None:
                i2_ = old_to_new_idxs[i2]
                i1_ = old_to_new_idxs[i1]
                between_factors.add(gtsam.BetweenFactorRot3(i2_, i1_, i2Ri1, noise_model))

        return between_factors

    def __between_factors_from_2view_relative_rotations(
        self, i2Ri1_dict: Dict[Tuple[int, int], Rot3], old_to_new_idxs: Dict[int, int]
    ) -> BetweenFactorPose3s:
        """Create between factors from relative rotations computed by the 2-view estimator."""
        # TODO: how to weight the noise model on relative rotations compared to priors?
        noise_model = gtsam.noiseModel.Isotropic.Sigma(POSE3_DOF, self._two_view_rotation_sigma)

        between_factors = BetweenFactorPose3s()

        for (i1, i2), i2Ri1 in i2Ri1_dict.items():
            if i2Ri1 is not None:
                # ignore translation during rotation averaging
                i2Ti1 = Pose3(i2Ri1, np.zeros(3))
                i2_ = old_to_new_idxs[i2]
                i1_ = old_to_new_idxs[i1]
                between_factors.append(BetweenFactorPose3(i2_, i1_, i2Ti1, noise_model))

        return between_factors


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
            between_factors.append(BetweenFactorPose3(i2_, i1_, i1Ti2_prior.value, noise_model))

        return between_factors

    def _run_with_consecutive_ordering(
        self, 
        num_connected_nodes: int, 
        graph: gtsam.NonlinearFactorGraph, 
        between_factors: BetweenFactorPose3s, 
        initial: gtsam.Values,
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

        logger.info("Running GNC rotation averaging...")
        #shonan = ShonanAveraging3(between_factors, self.__get_shonan_params())
        #initial = shonan.initializeRandomly()

        optimizer = GncLMOptimizer(graph, initial, self.__get_gnc_params())
        result = optimizer.optimize()

        wRi_list_consecutive = [None] * num_connected_nodes
        for i in range(num_connected_nodes):
            if result.exists(i):
                wRi_list_consecutive[i] = result.atRot3(i)
        logger.info(wRi_list_consecutive)

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

        between_factors = self.__between_factors_from_2view_relative_rotations(
            i2Ri1_dict, old_to_new_idxes
        )

        initial = initialize_mst(num_images, i2Ri1_dict, corr_idxs, old_to_new_idxes)

        graph: gtsam.NonlinearFactorGraph = self.__graph_from_2view_relative_rotations(
            i2Ri1_dict, old_to_new_idxes
        )
        # between_factors.extend(self._between_factors_from_pose_priors(i1Ti2_priors, old_to_new_idxes))

        wRi_list_subset = self._run_with_consecutive_ordering(
            len(nodes_with_edges), graph, between_factors, initial
        )

        wRi_list = [None] * num_images
        for remapped_i, original_i in enumerate(nodes_with_edges):
            wRi_list[original_i] = wRi_list_subset[remapped_i]

        return wRi_list


def initialize_mst(
        num_images: int, 
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]], 
        corr_idxs: Dict[Tuple[int, int], np.ndarray],
        old_to_new_idxs: Dict[int, int],
    ) -> gtsam.Values:
    """Initialize global rotations using the minimum spanning tree (MST)."""
    # Compute MST.
    row, col, data = [], [], []
    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None:
            continue
        row.append(i1)
        col.append(i2)
        data.append(-len(corr_idxs[(i1, i2)]))
    corr_adjacency = scipy.sparse.coo_array((data, (row, col)), shape=(num_images, num_images))
    Tcsr = scipy.sparse.csgraph.minimum_spanning_tree(corr_adjacency)
    logger.info(Tcsr.toarray().astype(int))

    # Build global rotations from MST.
    i_mst, j_mst = Tcsr.nonzero()
    logger.info(i_mst)
    logger.info(j_mst)
    edges_mst = [(i, j) for (i, j) in zip(i_mst, j_mst)]
    iR0_dict = {i_mst[0]: np.eye(3)}  # pick the left index of the first edge as the seed
    # max_iters = num_images * 10
    iter = 0
    while len(edges_mst) > 0:
        i, j = edges_mst.pop(0)
        if i in iR0_dict:
            jRi = i2Ri1_dict[(i, j)].matrix()
            iR0 = iR0_dict[i]
            iR0_dict[j] = jRi @ iR0
        elif j in iR0_dict:
            iRj = i2Ri1_dict[(i, j)].matrix().T
            jR0 = iR0_dict[j]
            iR0_dict[i] = iRj @ jR0
        else:
            edges_mst.append((i, j))
        iter += 1
        # if iter >= max_iters:
        #     logger.info("Reached max MST iters.")
        #     assert False
    
    # Add to Values object.
    initial = gtsam.Values()
    for i, iR0 in iR0_dict.items():
        initial.insert(old_to_new_idxs[i], Rot3(iR0))
    
    return initial



