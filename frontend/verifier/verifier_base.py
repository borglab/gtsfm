"""
Base class for the V (verification) stage of the frontend.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Tuple

import dask
import numpy as np


class VerifierBase(metaclass=abc.ABCMeta):
    """Base class for all verifiers, the final stage of the front-end.

    Verifiers take the coordinates of the matches as inputs and returns the 
    estimated fundamental matrix as well as geometrically verified points.
    """

    def __init__(self, min_pts):
        self.min_pts = min_pts

    @abc.abstractmethod
    def verify(self,
               matched_features_im1: np.ndarray,
               matched_features_im2: np.ndarray,
               image_shape_im1: Tuple[int, int],
               image_shape_im2: Tuple[int, int],
               camera_instrinsics_im1: np.ndarray = None,
               camera_instrinsics_im2: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Perform the geometric verification of the matched features.

        Note:
        1. The number of input features from image #1 and image #2 are equal.
        2. The function computes the fundamental matrix if intrinsics are not
            provided. Otherwise, it computes the essential matrix.

        Args:
            matched_features_im1 (np.ndarray): matched features from image #1
            matched_features_im2 (np.ndarray): matched features from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2
            camera_instrinsics_im1 (np.ndarray, optional): Camera intrinsics
                matrix for image #1. Defaults to None.
            camera_instrinsics_im2 (np.ndarray, optional): Camera intrinsics
                matris for image #2. Default to None

        Returns:
            np.ndarray: estimated fundamental/essential matrix
            np.ndarray: index of the match features which are verified
        """

    def verify_and_get_features(
            self,
            matched_features_im1: np.ndarray,
            matched_features_im2: np.ndarray,
            image_shape_im1: Tuple[int, int],
            image_shape_im2: Tuple[int, int],
            camera_instrinsics_im1: np.ndarray = None,
            camera_instrinsics_im2: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform the geometric verification of the matched features and 
        return the F-matrix and actual verified features.

        Note:
        1. The number of input features from image #1 are the same as the number from image #2
        2. The function computes the fundamental matrix if intrinsics are not
            provided. Otherwise, it computes the essential matrix.

        Args:
            matched_features_im1 (np.ndarray): matched features from image #1
            matched_features_im2 (np.ndarray): matched features from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2
            camera_instrinsics_im1 (np.ndarray, optional): Camera intrinsics
                matrix for image #1. Defaults to None.
            camera_instrinsics_im2 (np.ndarray, optional): Camera intrinsics
                matris for image #2. Default to None

        Returns:
            np.ndarray: tuple with three numpy arrays
                1. estimated fundamental matrix
                2. verified feature from image #1
                3. corresponding verified feature from image #2
        """
        geometry, verified_indices = self.verify(
            matched_features_im1,
            matched_features_im2,
            image_shape_im1,
            image_shape_im2,
            camera_instrinsics_im1,
            camera_instrinsics_im2
        )

        return geometry, matched_features_im1[verified_indices], matched_features_im2[verified_indices]

    def create_computation_graph(self,
                                 matcher_graph: Dict[Tuple[int, int], dask.delayed],
                                 image_shapes: List[Tuple[int, int]],
                                 camera_instrinsics: List[np.ndarray] = None
                                 ) -> Dict[Tuple[int, int], dask.delayed]:
        """Created the computation graph for verification using the graph 
        from matcher stage

        Args:
            matcher_graph (Dict[Tuple[int, int], dask.delayed]): computation
                graph from matcher
            image_shapes (List[Tuple[int, int]]): list of all image shapes
            camera_instrinsics_im1 (List[np.ndarray], optional): camera
                intrinsics matrix all images. Defaults to empty list.

        Returns:
            Dict[Tuple[int, int], dask.delayed]: delayed dask tasks for
                verification
        """

        result = dict()

        def camera_intrinsics_fetcher(
                idx):
            return None if camera_instrinsics is None else camera_instrinsics[idx]

        for image_idx_tuple, delayed_matcher in matcher_graph.items():
            result[image_idx_tuple] = dask.delayed(self.verify_and_get_features)(
                delayed_matcher[0],
                delayed_matcher[1],
                image_shapes[image_idx_tuple[0]],
                image_shapes[image_idx_tuple[1]],
                camera_intrinsics_fetcher(image_idx_tuple[0]),
                camera_intrinsics_fetcher(image_idx_tuple[1]),
            )

        return result
