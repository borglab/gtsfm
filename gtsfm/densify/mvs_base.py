"""Base class for the multi-view stereo (MVS) stage of the back-end.

Authors: John Lambert, Ren Liu
"""
import abc
from typing import Dict

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import PinholeCameraCal3Bundler

from gtsfm.common.image import Image


class MVSBase(metaclass=abc.ABCMeta):
    """Base class for all multi-view stereo implementations."""

    def __init__(self) -> None:
        """Initialize the MVS module """
        pass

    @abc.abstractmethod
    def densify(
        self,
        images: Dict[int,Image],
        cameras: Dict[int, PinholeCameraCal3Bundler],
        min_distance: float,
        max_distance: float,
    ) -> np.ndarray:
        """Densify a point cloud using multi-view stereo.
        
        Note: we do not return depth maps here per image, as they would need to be aligned to ground truth
        before evaluation for completeness, etc.

        Args:
            images: dictionary mapping image indices to input images.
            cameras: dictionary mapping image indices to camera parameters
            min_distance: minimum distance from any camera to any 3d point
            max_distance: maximum distance from any camera to any 3d point

        Returns:
            Dense point cloud, as an array of shape (N,3)
        """

    def create_computation_graph(
        self,
        images_graph: Delayed,
        cameras_graph: Delayed,
        min_distance_graph: Delayed,
        max_distance_graph: Delayed
    ) -> Delayed:
        """Generates the computation graph for performing multi-view stereo.

        Args:
            images_graph: computation graph for images.
            cameras_graph: computation graph for cameras
            min_distance_graph: minimum distance from any camera to any 3d point, wrapped up as Delayed
            max_distance_graph: minimum distance from any camera to any 3d point, wrapped up as Delayed

        Returns:
            Delayed task for MVS computation on the input images.
        """
        return dask.delayed(self.densify)(images_graph, cameras_graph, min_distance_graph, max_distance_graph)
