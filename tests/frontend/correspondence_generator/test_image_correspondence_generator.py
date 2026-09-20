"""Tests for ImageCorrespondenceGenerator constructor options."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_dedup import (
    KeypointAggregatorDedup,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)


class TestImageCorrespondenceGeneratorInit(unittest.TestCase):
    def test_default_uses_dedup_aggregator(self) -> None:
        gen = ImageCorrespondenceGenerator(matcher=MagicMock())
        self.assertIsInstance(gen._aggregator, KeypointAggregatorDedup)

    def test_deduplicate_false_uses_unique_aggregator(self) -> None:
        gen = ImageCorrespondenceGenerator(matcher=MagicMock(), deduplicate=False)
        self.assertIsInstance(gen._aggregator, KeypointAggregatorUnique)

    def test_explicit_aggregator_overrides_deduplicate(self) -> None:
        aggregator = KeypointAggregatorDedup(nms_merge_radius=1e-4)
        gen = ImageCorrespondenceGenerator(matcher=MagicMock(), aggregator=aggregator, deduplicate=False)
        self.assertIs(gen._aggregator, aggregator)
        self.assertEqual(gen._aggregator.nms_merge_radius, 1e-4)


if __name__ == "__main__":
    unittest.main()
