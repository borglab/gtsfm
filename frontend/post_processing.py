"""Post processer for the front-end, to transform the front-end output into 
inputs for downstream components

Authors: Ayush Baid
"""
from typing import Dict, List, Tuple, Union

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Rot3, Unit3

import utils.verification as verification_utils


class FrontEndPostProcessor:

    def run(self,
            geometry: np.ndarray,
            verified_correspondences_im1: np.ndarray,
            verified_correspondences_im2: np.ndarray,
            camera_intrinsics_im1: np.ndarray,
            camera_intrinsics_im2: np.ndarray,
            is_essential: bool = True
            ) -> Tuple[Union[Rot3, None], Union[Unit3, None]]:
        """Run the post-processing on the front-end function.

        Args:
            geometry: Fundamental matrix/essential matrix from front-end.
            verified_correspondences_im1: verified feature matches from im1.
            verified_correspondences_im2: verified feature matches from im2.
            camera_intrinsics_im1: intrinsics for im1.
            camera_intrinsics_im2: intrinsics from im2
            is_essential: Boolean flag indicating if the geometry is essential 
                          matrix. Defaults to True.

        Returns:
            Union[Rot3, None]: recovered rotation from im1 to im2.
            Union[Unit3, None]: recovered unit translation from im1 to im2.
        """
        if is_essential:
            return verification_utils.recover_pose_from_essential_matrix(
                (geometry, verified_correspondences_im1,
                 verified_correspondences_im2),
                camera_intrinsics_im1,
                camera_intrinsics_im2
            )
        else:
            return verification_utils.recover_pose_from_fundamental_matrix(
                (geometry, verified_correspondences_im1,
                 verified_correspondences_im2),
                camera_intrinsics_im1,
                camera_intrinsics_im2
            )

    def create_computation_graph(self,
                                 frontend_graph: Dict[Tuple[int, int], Delayed],
                                 intrinsics_graph: List[Delayed],
                                 is_essential: bool = True
                                 ) -> Tuple[
                                     Dict[Tuple[int, int], Delayed],
                                     Dict[Tuple[int, int], Delayed]]:
        """Creates computation graph to recover relative rotations and relative
        translations from frontend output.

        Args:
            frontend_graph: computation graph with output from the front-end.
            intrinsics_graph: graph with getters for camera intrinsics.
            is_essential: Boolean flag indicating if the geometry is essential 
                          matrix. Defaults to True.

        Returns:
            Dict[Tuple[int, int], Delayed]: graph for relative rotations.
            Dict[Tuple[int, int], Delayed]: graph for relative translations.
        """

        i1Ri2_graph = dict()
        i1ti2_graph = dict()

        for pose_idx_i1i2, frontend_graph_element in frontend_graph.items():

            delayed_element = dask.delayed(self.run)(
                frontend_graph_element[0],
                frontend_graph_element[1],
                frontend_graph_element[2],
                intrinsics_graph[pose_idx_i1i2[0]],
                intrinsics_graph[pose_idx_i1i2[1]],
                is_essential
            )

            i1Ri2_graph[pose_idx_i1i2] = delayed_element[0]
            i1ti2_graph[pose_idx_i1i2] = delayed_element[1]

        return i1Ri2_graph, i1ti2_graph
