"""Base class for the gaussian splatting (GS) stage of the back-end.

Authors: Harneet Singh Khanuja
"""

import abc
from typing import Mapping, Tuple

import dask
from dask.delayed import Delayed

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class GSBase(GTSFMProcess):
    """Base class for gaussian_splatting implementations."""

    def get_ui_metadata() -> UiMetadata:  # type: ignore
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="3D Gaussian Splatting",
            input_products=("Images", "Optimized Camera Poses", "Optimized 3D Tracks"),
            output_products=("Dense Colored 3D Point Cloud",),
            parent_plate="Gaussian Splatting",
        )

    def __init__(self) -> None:
        """Initialize the Gaussian Splatting module"""
        pass

    @abc.abstractmethod
    def splatify(
        self,
        images_by_index: Mapping[int, Image],
        sfm_result_graph: GtsfmData,
    ) -> Tuple[object, object]:
        """Create 3D gaussians using Gaussian Splatting.

        Args:
            images_by_index: Mapping of image index to image object.
            sfm_result_graph: object containing camera parameters and the optimized point cloud.

        Returns:
            gaussian_splats: 3D Gaussian Splats defining the entire scene
            cfg: Config class object which will be used while rendering
        """
        raise NotImplementedError

    def create_computation_graph(self, images_graph: Delayed, sfm_result_graph: Delayed) -> Tuple[Delayed, Delayed]:
        """Generates the computation graph for performing the gaussian splats and the config parameters.

        Args:
            images_graph: computation graph for index→image mapping.
            sfm_result_graph: computation graph for SFM output

        Returns:
            Delayed task for splats computation on the input images
        """
        splats_graph, cfg_graph = dask.delayed(self.splatify, nout=2)(images_graph, sfm_result_graph)  # type: ignore

        return splats_graph, cfg_graph
