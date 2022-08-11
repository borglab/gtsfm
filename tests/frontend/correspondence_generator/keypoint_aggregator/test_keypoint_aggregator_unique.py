"""Tests for frontend's DoG detector class.

Authors: Ayush Baid
"""
import unittest

from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)
from tests.frontend.correspondence_generator.keypoint_aggregator import test_keypoint_aggregator_base


class TestKeypointAggregatorUnique(test_keypoint_aggregator_base.TestKeypointAggregatorBase):
    """Test class for DoG detector class in frontend.

    All unit test functions defined in TestDetectorBase are run automatically.
    """

    def setUp(self):
        super().setUp()
        self.aggregator = KeypointAggregatorUnique()


if __name__ == "__main__":
    unittest.main()
