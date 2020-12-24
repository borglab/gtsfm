"""Tests for the two-way matcher.

Authors: Ayush Baid
"""
import unittest
import numpy as np

import tests.frontend.matcher.test_matcher_base as test_matcher_base
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher


class TestTwoWayMatcher(test_matcher_base.TestMatcherBase):
    """Unit tests for the TwoWayMatcher.

    All unit test functions defined in TestMatcherBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.matcher = TwoWayMatcher()

    def test_on_dummy_data(self):
        """Test using dummy 1D descriptors to verify correctness."""

        desc1 = np.array([0.4865,
                          0.3752,
                          0.3077,
                          0.9188,
                          0.7837,
                          0.1083,
                          0.6822,
                          0.3764,
                          0.2288,
                          0.8018,
                          1.1]
                         ).reshape(-1, 1).astype(np.float32)

        desc2 = np.array([0.9995,
                          0.3376,
                          0.9005,
                          0.5382,
                          0.3162,
                          0.7974,
                          0.1785,
                          0.3491,
                          0.8658,
                          0.2912]
                         ).reshape(-1, 1).astype(np.float32)

        result = self.matcher.match(desc1, desc2)

        np.testing.assert_array_equal(np.array([
            [9, 5],
            [2, 4],
            [3, 2],
            [1, 7],
            [8, 6],
            [0, 3]
        ]), result)


if __name__ == '__main__':
    unittest.main()
