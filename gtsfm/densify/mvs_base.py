"""Base class for the multi-view stereo (MVS) stage of the back-end.

Authors: John Lambert, Ren Liu
"""
import abc
from typing import Dict, Tuple

import dask
import numpy as np
from dask.delayed import Delayed

import gtsfm.densify.mvs_utils as mvs_utils
from gtsfm.common.image import Image
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetricsGroup


class MVSBase(metaclass=abc.ABCMeta):
    """Base class for all multi-view stereo implementations."""

    def __init__(self) -> None:
        """Initialize the MVS module"""
        pass

    @abc.abstractmethod
    def densify(
        self, images: Dict[int, Image], sfm_result: GtsfmData
    ) -> Tuple[np.ndarray, np.ndarray, GtsfmMetricsGroup]:
        """Densify a point cloud using multi-view stereo.

        Note: we do not return depth maps here per image, as they would need to be aligned to ground truth
        before evaluation for completeness, etc.

        Args:
            images: dictionary mapping image indices to input images.
            sfm_result: object containing camera parameters and the optimized point cloud.
                We can use this point cloud to determine the min and max depth (from any
                camera to any 3d point) for plane-sweeping stereo.

        Returns:
            dense_points: 3D coordinates (in the world frame) of the dense point cloud
                with shape (N, 3) where N is the number of points
            dense_point_colors: RGB color of each point in the dense point cloud
                with shape (N, 3) where N is the number of points
            densify_metrics: Metrics group containing metrics for dense reconstruction
        """

    def create_computation_graph(
        self, images_graph: Dict[int, Delayed], sfm_result_graph: Delayed
    ) -> Tuple[Delayed, Delayed, Delayed, Delayed]:
        """Generates the computation graph for performing multi-view stereo.

        Args:
            images_graph: computation graph for images.
            sfm_result_graph: computation graph for SFM output

        Returns:
            Delayed tasks for MVS computation on the input images, including:
                1. downsampled dense point cloud
                2. rgb colors for each point in the point cloud
                3. mvs densify metrics group
                4. voxel downsampling metrics group
        """
        # get initial dense reconstruction result
        points_graph, rgb_graph, densify_metrics_graph = dask.delayed(self.densify, nout=3)(
            images_graph, sfm_result_graph
        )

        # calculate the scale of target occupied volume, then compute the minimum voxel size for downsampling
        voxel_size_graph = dask.delayed(mvs_utils.estimate_minimum_voxel_size, nout=1)(points_graph)

        # downsampling the volume to avoid duplicates in the point cloud
        sampled_points_graph, sampled_rgb_graph = dask.delayed(mvs_utils.downsample_point_cloud, nout=2)(
            points_graph, rgb_graph, voxel_size_graph
        )

        # calculate downsampling metrics
        downsampling_metrics_graph = dask.delayed(mvs_utils.get_voxel_downsampling_metrics, nout=1)(
            voxel_size_graph, points_graph, sampled_points_graph
        )
        return sampled_points_graph, sampled_rgb_graph, densify_metrics_graph, downsampling_metrics_graph
