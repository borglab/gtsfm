"""Base class for front end matchers that directly identify image correspondences, without using explicit descriptors.

Authors: John Lambert
"""
import abc
from typing import Optional, Tuple

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class ImageMatcherBase(GTSFMProcess):
    """Base class for matchers that accept an image pair, and directly generate keypoint matches.

    Note: these matchers do NOT use descriptors as input.
    """

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="Direct Image Matcher",
            input_products=("Images",),
            output_products=("Detections i", "Detections j"),
            parent_plate="ImageCorrespondenceGenerator",
        )

    @abc.abstractmethod
    def match(
        self,
        image_i1: Image,
        image_i2: Image,
        keypoints_i1: Optional[Keypoints] = None,
        keypoints_i2: Optional[Keypoints] = None,
    ) -> Tuple[Keypoints, Keypoints]:
        """Identify feature matches across two images.

        Args:
            image_i1: first input image of pair.
            image_i2: second input image of pair.
            keypoints_i1: precomputed keypoints in first image with which to query dense warp, if applicable.
            keypoints_i2: precomputed keypoints in second image with which to query dense warp, if applicable.

        Returns:
            Keypoints from image 1 (N keypoints will exist).
            Corresponding keypoints from image 2 (there will also be N keypoints). These represent feature matches.
        """
