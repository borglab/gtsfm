"""Base class for the multi-view stereo (MVS) stage of the back-end.

Authors: John Lambert, Ren Liu
"""
import abc
from typing import Dict, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed

import gtsfm.densify.mvs_utils as mvs_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.outputs import Outputs
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class MVSBase(GTSFMProcess):
    """Base class for all multi-view stereo implementations."""

    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="Multi-view Stereo",
            input_products=("Images", "Optimized Camera Poses", "Optimized 3D Tracks"),
            output_products="Dense Colored 3D Point Cloud",
            parent_plate="Dense Reconstruction",
        )

    def __init__(self) -> None:
        """Initialize the MVS module"""
        pass

    @abc.abstractmethod
    def densify(
        self, images: Dict[int, Image], sfm_result: GtsfmData, outputs: Optional[Outputs] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        self, images_graph: Delayed, sfm_result_graph: Delayed, outputs: Optional[Outputs] = None
    ) -> Tuple[Delayed, Delayed, Optional[Delayed]]:
        """Generates the computation graph for performing multi-view stereo.

        Args:
            images_graph: computation graph for images.
            sfm_result_graph: computation graph for SFM output

        Returns:
            Tuple containing delayed tasks for MVS computation on the input images:
                1. downsampled dense point cloud
                2. rgb colors for each point in the point cloud
                3. optional delayed task that persists metrics (None when metrics are disabled)
        """
        # get initial dense reconstruction result
        points_graph, rgb_graph = dask.delayed(self.densify, nout=2)(
            images_graph, sfm_result_graph, outputs
        )

        # calculate the scale of target occupied volume, then compute the minimum voxel size for downsampling
        voxel_size_graph = dask.delayed(mvs_utils.estimate_minimum_voxel_size, nout=1)(points_graph)

        # downsampling the volume to avoid duplicates in the point cloud
        sampled_points_graph, sampled_rgb_graph = dask.delayed(mvs_utils.downsample_point_cloud, nout=2)(
            points_graph, rgb_graph, voxel_size_graph
        )

        # calculate downsampling metrics if a sink is available
        metrics_task: Optional[Delayed] = None
        if outputs is not None and outputs.metrics_sink is not None:
            metrics_task = dask.delayed(mvs_utils.get_voxel_downsampling_metrics)(
                voxel_size_graph, points_graph, sampled_points_graph, outputs.metrics_sink
            )
        return sampled_points_graph, sampled_rgb_graph, metrics_task
