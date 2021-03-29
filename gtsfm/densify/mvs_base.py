"""Base class for the MVS stage of the back-end.

Authors: John Lambert, Ren Liu
"""
import abc
from typing import Dict, Tuple

import dask
from dask.delayed import Delayed
from gtsam import PinholeCameraCal3Bundler

from gtsfm.common.image import Image
from gtsfm.common.gtsfm_data import GtsfmData

class MvsBase(metaclass=abc.ABCMeta):
    """Base class for all multi-view stereo hyperparameters."""

    def __init__(self) -> None:
        """Initialize the MVS module """
        pass

    @abc.abstractmethod
    def densify(
        self,
        images: Dict[int,Image],
        cameras: Dict[int, PinholeCameraCal3Bundler],
        gtsfm_data: Optional[GtsfmData] = None,
        min_distance: Optional[float] = None,
        max_distance: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[int,np.ndarray]]:
        """Densify a point cloud using multi-view stereo

        Args:
            image: dictionary mapping image indices to input images.
            cameras: dictionary mapping image indices to camera parameters
            gtsfm_data: TODO 
            min_distance: minimum distance from any camera to any 3d point
            max_distance: maximum distance from any camera to any 3d point

        Returns:
            (N,3) representing dense point cloud
            dictionary mapping integer index to depth map of shape (H,W)
        """

    def create_computation_graph(self, images_graph: Delayed, cameras_graph: Delayed) -> Delayed:
        """Generates the computation graph for performing multi-view stereo.

        Args:
            images_graph: computation graph for images.
            cameras_graph

        Returns:
            Delayed task for detection on the input image.
        """
        return dask.delayed(self.densify)(image_graph, cameras_graph)
