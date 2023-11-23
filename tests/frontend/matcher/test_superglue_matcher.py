"""Tests for the SuperGlue matcher.

Authors: John Lambert
"""
import unittest

import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.superglue_matcher import SuperGlueMatcher


class TestSuperGlueMatcher(unittest.TestCase):
    """Unit tests for the SuperGlueMatcher.

    All unit test functions defined in TestMatcherBase are run automatically.
    """

    def setUp(self) -> None:
        super().setUp()

        self.matcher = SuperGlueMatcher()

    def test_on_dummy_data(self) -> None:
        """Ensure the SuperGlue matcher returns output of the correct shape, for random input."""
        # image height and width
        H, W, C = 20, 20, 3

        im_shape_i1 = (H, W, C)
        im_shape_i2 = (H, W, C)

        num_kps_i1 = 50
        kps_i1 = Keypoints(coordinates=np.random.randint(0, H, size=(num_kps_i1, 2)), responses=np.random.rand(50))
        descs_i1 = np.random.randn(num_kps_i1, 256)

        num_kps_i2 = 100
        kps_i2 = Keypoints(coordinates=np.random.randint(0, H, size=(num_kps_i2, 2)), responses=np.random.rand(100))
        descs_i2 = np.random.randn(num_kps_i2, 256)

        match_indices = self.matcher.match(kps_i1, kps_i2, descs_i1, descs_i2, im_shape_i1, im_shape_i2)
        assert isinstance(match_indices, np.ndarray)
        assert match_indices.dtype == np.uint32
