"""Unit tests for the image data structure.

Authors: Ayush Baid
"""
import unittest
import unittest.mock as mock

import numpy as np
from gtsam import Cal3Bundler

from common.image import Image
from utils.sensor_width_database import SensorWidthDatabase


class TestImage(unittest.TestCase):
    """Unit tests for the image class."""

    @mock.patch.object(SensorWidthDatabase, '__init__', return_value=None)
    @mock.patch.object(SensorWidthDatabase, 'lookup', return_value=5)
    def test_get_intrinsics_from_exif(self, mock_init, mock_lookup):
        """Tests the intrinsics generation frome exif."""

        exif_data = {
            'FocalLength': 25,
            'Make': 'testMake',
            'Model': 'testModel',
        }

        expected_instrinsics = Cal3Bundler(
            fx=600.0, k1=0.0, k2=0.0, u0=50.0, v0=60.0)

        image = Image(np.random.randint(low=0, high=255, size=(100, 120, 3)),
                      exif_data)

        computed_intrinsics = image.get_intrinsics_from_exif()

        self.assertTrue(expected_instrinsics.equals(computed_intrinsics, 1e-3))
