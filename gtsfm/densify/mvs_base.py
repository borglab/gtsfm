"""Base class for the multi-view stereo (MVS) stage of the back-end.

Authors: John Lambert, Ren Liu
"""
import abc
from typing import Dict, Tuple

import numpy as np

import gtsfm.densify.mvs_utils as mvs_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class MVSBase(GTSFMProcess):
    """Base class for all multi-view stereo implementations."""

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="Multi-view Stereo",
            input_products=(
                "Images",
                "Optimized Camera Poses",
                "Optimized 3D Tracks",
            ),
            output_products="Dense Colored 3D Point Cloud",
            parent_plate="Dense Reconstruction",
        )

    def __init__(self) -> None:
        """Initialize the MVS module"""
        pass

    @abc.abstractmethod
    def apply(
        self, images: Dict[int, Image], sfm_result: GtsfmData
    ) -> Tuple[np.ndarray, np.ndarray, GtsfmMetricsGroup, GtsfmMetricsGroup]:
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
            downsamping_metrics: Metrics group for downsampling metrics.
        """

    def compute_downsampling_metrics(self, points: np.ndarray, colors: np.ndarray):
        # calculate the scale of target occupied volume, then compute the minimum voxel size for downsampling
        voxel_size = mvs_utils.estimate_minimum_voxel_size(points)

        # downsampling the volume to avoid duplicates in the point cloud
        sampled_points, _ = mvs_utils.downsample_point_cloud(points, colors, voxel_size)

        downsampling_metrics = mvs_utils.get_voxel_downsampling_metrics(voxel_size, points, sampled_points)

        return downsampling_metrics
