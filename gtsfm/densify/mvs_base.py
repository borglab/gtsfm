"""Base class for the multi-view stereo (MVS) stage of the back-end.

Authors: John Lambert, Ren Liu
"""
import abc
from typing import Dict

import dask
import numpy as np
from dask.delayed import Delayed

from gtsfm.common.image import Image
from gtsfm.common.gtsfm_data import GtsfmData


class MVSBase(metaclass=abc.ABCMeta):
    """Base class for all multi-view stereo implementations."""

    def __init__(self) -> None:
        """Initialize the MVS module """
        pass

    @abc.abstractmethod
    def densify(self, images: Dict[int, Image], sfm_result: GtsfmData) -> np.ndarray:
        """Densify a point cloud using multi-view stereo.

        Note: we do not return depth maps here per image, as they would need to be aligned to ground truth
        before evaluation for completeness, etc.

        Args:
            images: dictionary mapping image indices to input images.
            sfm_result: object containing camera parameters and the optimized point cloud.
                We can use this point cloud to determine the min and max depth (from any
                camera to any 3d point) for plane-sweeping stereo.

        Returns:
            Dense point cloud, as an array of shape (N,3)
        """

    def create_computation_graph(self, images_graph: Delayed, sfm_result_graph: Delayed) -> Delayed:
        """Generates the computation graph for performing multi-view stereo.

        Args:
            images_graph: computation graph for images.
            sfm_result_graph: computation graph for SFM output

        Returns:
            Delayed task for MVS computation on the input images.
        """
        return dask.delayed(self.densify)(images_graph, sfm_result_graph)
