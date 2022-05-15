"""Extracts features and their descriptors from a single image.

Authors: Ayush Baid, John Lambert
"""
from typing import Dict, Tuple

from dask.delayed import Delayed

from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


class FeatureExtractor:
    """Wrapper for running detection and description on each image."""

    def __init__(self, detector_descriptor: DetectorDescriptorBase) -> None:
        """Initializes the detector-descriptor

        Args:
            detector_descriptor: the joint detector-descriptor to use.
        """
        self.detector_descriptor = detector_descriptor

    def create_computation_graph(self, delayed_images: Dict[int, Delayed]) -> Tuple[Delayed, Delayed]:
        """Given an image, create detection and descriptor generation tasks

        Args:
            delayed_images: image wrapped up in Delayed

        Returns:
            Delayed object for detected keypoints.
            Delayed object for corr. descriptors.
        """
        return self.detector_descriptor.create_computation_graph(delayed_images)
