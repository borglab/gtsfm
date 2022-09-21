"""Joint detector and descriptor for the front end.

Authors: Ayush Baid
"""

import abc
from typing import Tuple

import dask
import numpy as np
from dask.delayed import Delayed

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata

logger = logger_utils.get_logger()


class DetectorDescriptorBase(GTSFMProcess):
    """Base class for all methods which provide a joint detector-descriptor to work on an image.

    This class serves as a combination of individual detector and descriptor.
    """

    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="DetectorDescriptor",
            input_products="Images",
            output_products=("Keypoints", "Descriptors"),
            parent_plate="DetDescCorrespondenceGenerator",
        )

    def __init__(self, max_keypoints: int = 5000):
        """Initialize the detector-descriptor.

        Args:
            max_keypoints: Maximum number of keypoints to detect. Defaults to 5000.
        """
        self.max_keypoints = max_keypoints

    @abc.abstractmethod
    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for
        details about the output format.

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """

    def create_computation_graph(self, image_graph: Delayed) -> Tuple[Delayed, Delayed]:
        """Generates the computation graph for detections and their descriptors.

        Args:
            image_graph: computation graph for a single image (from a loader).

        Returns:
            Delayed tasks for detections.
            Delayed task for corr. descriptors.
        """
        keypoints_graph, descriptor_graph = dask.delayed(self.detect_and_describe, nout=2)(image_graph)

        return keypoints_graph, descriptor_graph
