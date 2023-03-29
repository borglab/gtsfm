"""Correspondence generator that utilizes explicit keypoint detection, following by descriptor matching, per image.

Authors: John Lambert
"""
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.matcher.matcher_base import MatcherBase


class DetDescCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Traditional pair-wise matching of descriptors."""

    def __init__(self, matcher: MatcherBase, det_desc: DetectorDescriptorBase):
        """
        Args:
            matcher: matcher to use.
            feature_extractor: feature extractor to use.
        """
        self.det_desc = det_desc
        self.matcher = matcher
