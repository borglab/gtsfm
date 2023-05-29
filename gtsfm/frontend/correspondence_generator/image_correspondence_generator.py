"""Correspondence generator that utilizes direct matching of keypoints across an image pair, without descriptors.

Authors: John Lambert
"""
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_dedup import (
    KeypointAggregatorDedup,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_base import KeypointAggregatorBase
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase


class ImageCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Pair-wise direct matching of images (e.g. transformer-based)."""

    def __init__(self, matcher: ImageMatcherBase, deduplicate: bool = True) -> None:
        """
        Args:
            matcher: matcher to use.
            deduplicate: whether to de-duplicate with a single image the detections received from each image pair.
        """
        self.matcher = matcher

        self.aggregator: KeypointAggregatorBase = (
            KeypointAggregatorDedup() if deduplicate else KeypointAggregatorUnique()
        )
