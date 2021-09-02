"""Tests for frontend's image retriever base class.

Authors: Travis Driver
"""

import unittest

from gtsfm.frontend.retriever.retriever_base import RetrieverBase
from gtsfm.two_view_estimator import TwoViewEstimator


class TestRetrieverBase(unittest.TestCase):
    """Test class."""

    def setUp(self):
        super().setUp()

        self.image_retriever = RetrieverBase()
