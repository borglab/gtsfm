"""Extracts features and their descriptors from a single image.

Authors: Ayush Baid, John Lambert
"""
import logging
import sys
from typing import Tuple

from dask.delayed import Delayed
from gtsam import (
    Cal3Bundler,
    PinholeCameraCal3Bundler,
    Pose3,
    Rot3,
    SfmData,
    Unit3,
)

import gtsfm.utils.serialization  # import needed to register serialization fns
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import (
    DetectorDescriptorBase,
)

# configure loggers to avoid DEBUG level stdout messages
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


class FeatureExtractor:
    """Wrapper for running detection and description on each image."""

    def __init__(self, detector_descriptor: DetectorDescriptorBase) -> None:
        """Initializes the detector-descriptor

        Args:
            detector_descriptor: the joint detector-descriptor to use.
        """
        self.detector_descriptor = detector_descriptor

    def create_computation_graph(
        self, image_graph: Delayed
    ) -> Tuple[Delayed, Delayed]:
        """Given an image, create detection and descriptor generation tasks

        Args:
            image_graph: image wrapped up in Delayed

        Returns:
            Delayed object for detected keypoints.
            Delayed object for corr. descriptors.
        """
        return self.detector_descriptor.create_computation_graph(image_graph)
