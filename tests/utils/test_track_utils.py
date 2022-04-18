"""Unit tests for track utility functions.

Authors: Ayush Baid
"""
import unittest
from unittest import mock

import gtsfm.utils.tracks as track_utils


class TestTrackUtils(unittest.TestCase):
    """Unit tests for the track utils."""

    @mock.patch("gtsfm.data_association.point3d_initializer.Point3dInitializer")
    def test_classify_tracks2d_with_gt_cameras(self, mock_point3d_initializer):
        mock_point3d_initializer.return_value.triangulate.return_value = None
