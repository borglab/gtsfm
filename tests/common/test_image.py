"""Unit tests for the image data structure.

Authors: Ayush Baid
"""
import unittest
import unittest.mock as mock

import numpy as np
from gtsam import Cal3Bundler

from gtsfm.common.image import Image
from gtsfm.common.sensor_width_database import SensorWidthDatabase

DEFAULT_FOCAL_LENGTH_FACTOR = 1.2


class TestImage(unittest.TestCase):
    """Unit tests for the image class."""

    def test_get_intrinsics_no_exif(self):
        """Tests the intrinsics generation from exif."""

        im_h = 100
        im_w = 120
        exif_data = None
        image = Image(np.random.randint(low=0, high=255, size=(im_h, im_w, 3)), exif_data)
        computed_intrinsics = image.get_intrinsics(default_focal_length_factor=DEFAULT_FOCAL_LENGTH_FACTOR)

        expected_focal_length = DEFAULT_FOCAL_LENGTH_FACTOR * max(im_h, im_w)
        expected_intrinsics = Cal3Bundler(fx=expected_focal_length, k1=0.0, k2=0.0, u0=60.0, v0=50.0)

        # Focal length default value is expected because exif data is None.
        self.assertTrue(expected_intrinsics.equals(computed_intrinsics, 1e-3))

    def test_get_intrinsics_no_tags(self):
        """Tests the intrinsics generation from exif."""

        im_h = 100
        im_w = 120
        exif_data = {
            "DummyName": "DummyValue",
        }
        image = Image(np.random.randint(low=0, high=255, size=(im_h, im_w, 3)), exif_data)
        computed_intrinsics = image.get_intrinsics(default_focal_length_factor=DEFAULT_FOCAL_LENGTH_FACTOR)

        expected_focal_length = DEFAULT_FOCAL_LENGTH_FACTOR * max(im_h, im_w)
        expected_intrinsics = Cal3Bundler(fx=expected_focal_length, k1=0.0, k2=0.0, u0=60.0, v0=50.0)

        # Focal length default value is expected because no tags of interest exist.
        self.assertTrue(expected_intrinsics.equals(computed_intrinsics, 1e-3))

    def test_get_intrinsics_from_exif_focal_length_in_35mm_film(self):
        """Tests the intrinsics generation from exif.

        Created a sample exif data below. The reading order is `FocalLengthIn35mmFilm`, `Make&Model`, `ExifImageWidth`.
        Based on the sample data, the focal length value computed from each case is 480.0, 600.0, 590.551.
        """

        im_h = 100
        im_w = 120
        exif_data = {
            "FocalLengthIn35mmFilm": 140,
            "FocalLength": 25,
            "Make": "testMake",
            "Model": "testModel",
            "ExifImageWidth": 100,
            "FocalPlaneXResolution": 1000,
            "FocalPlaneResolutionUnit": 2,
        }
        image = Image(np.random.randint(low=0, high=255, size=(im_h, im_w, 3)), exif_data)
        computed_intrinsics = image.get_intrinsics(default_focal_length_factor=DEFAULT_FOCAL_LENGTH_FACTOR)

        expected_intrinsics = Cal3Bundler(fx=480, k1=0.0, k2=0.0, u0=60.0, v0=50.0)

        # Focal length computed from `FocalLengthIn35mmFilm` is expected.
        self.assertTrue(expected_intrinsics.equals(computed_intrinsics, 1e-3))

    @mock.patch.object(SensorWidthDatabase, "__init__", return_value=None)
    @mock.patch.object(SensorWidthDatabase, "lookup", return_value=5)
    def test_get_intrinsics_from_exif_focal_length(self, mock_init, mock_lookup):
        """Tests the intrinsics generation from exif."""

        im_h = 100
        im_w = 120
        exif_data = {
            "FocalLength": 25,
            "Make": "testMake",
            "Model": "testModel",
            "ExifImageWidth": 100,
            "FocalPlaneXResolution": 1000,
            "FocalPlaneResolutionUnit": 2,
        }
        image = Image(np.random.randint(low=0, high=255, size=(im_h, im_w, 3)), exif_data)
        computed_intrinsics = image.get_intrinsics(default_focal_length_factor=DEFAULT_FOCAL_LENGTH_FACTOR)

        expected_intrinsics = Cal3Bundler(fx=600, k1=0.0, k2=0.0, u0=60.0, v0=50.0)

        # Focal length computed from `Make&Model` is expected.
        self.assertTrue(expected_intrinsics.equals(computed_intrinsics, 1e-3))

    def test_get_intrinsics_from_exif_exif_image_width(self):
        """Tests the intrinsics generation from exif."""

        im_h = 100
        im_w = 120
        exif_data = {
            "FocalLength": 25,
            "ExifImageWidth": 200,
            "FocalPlaneXResolution": 1000,
            "FocalPlaneResolutionUnit": 2,
        }
        image = Image(np.random.randint(low=0, high=255, size=(im_h, im_w, 3)), exif_data)
        computed_intrinsics = image.get_intrinsics(default_focal_length_factor=DEFAULT_FOCAL_LENGTH_FACTOR)

        expected_intrinsics = Cal3Bundler(fx=590.551, k1=0.0, k2=0.0, u0=60.0, v0=50.0)

        # Focal length computed from `ExifImageWidth` is expected.
        self.assertTrue(expected_intrinsics.equals(computed_intrinsics, 1e-3))

    def test_extract_patch_fully_inside(self):
        """Test patch extraction which is fully inside the original image."""

        input_image = Image(np.random.rand(100, 71, 3))

        patch_center_x = 21
        patch_center_y = 22

        """Test with even patch size."""
        patch_size = 10

        computed_patch = input_image.extract_patch(patch_center_x, patch_center_y, patch_size)

        # check the patch dimensions
        self.assertEqual(computed_patch.width, patch_size)
        self.assertEqual(computed_patch.height, patch_size)

        # check the patch contents
        np.testing.assert_allclose(
            computed_patch.value_array,
            input_image.value_array[
                patch_center_y - patch_size // 2 : patch_center_y + (patch_size + 1) // 2,
                patch_center_x - patch_size // 2 : patch_center_x + (patch_size + 1) // 2,
            ],
        )

        """Test with odd patch size."""
        patch_size = 11

        computed_patch = input_image.extract_patch(patch_center_x, patch_center_y, patch_size)

        # check the patch dimensions
        self.assertEqual(computed_patch.width, patch_size)
        self.assertEqual(computed_patch.height, patch_size)

        # check the patch contents
        np.testing.assert_allclose(
            computed_patch.value_array,
            input_image.value_array[
                patch_center_y - patch_size // 2 : patch_center_y + (patch_size + 1) // 2,
                patch_center_x - patch_size // 2 : patch_center_x + (patch_size + 1) // 2,
            ],
        )

    def test_extract_patch_for_padding(self):
        """Test patch extraction which is not completely inside the original image, hence needing padding."""

        input_image = Image(np.random.rand(100, 71, 3))

        patch_size = 10

        """Test with patch which hangs outside the image on the top left."""
        patch_center_x = 3
        patch_center_y = 4

        computed_patch = input_image.extract_patch(patch_center_x, patch_center_y, patch_size)

        # check the patch dimensions
        self.assertEqual(computed_patch.width, patch_size)
        self.assertEqual(computed_patch.height, patch_size)

        # check for zeros in area out of bounds
        np.testing.assert_allclose(computed_patch.value_array[:1, :], 0)
        np.testing.assert_allclose(computed_patch.value_array[:, :2], 0)

        # check the patch contents with the original input
        np.testing.assert_allclose(
            computed_patch.value_array[1:, 2:],
            input_image.value_array[
                : patch_center_y + (patch_size + 1) // 2,
                : patch_center_x + (patch_size + 1) // 2,
            ],
        )

        """Test with patch which hangs outside the image on bottom right."""
        patch_center_x = 70
        patch_center_y = 96

        computed_patch = input_image.extract_patch(patch_center_x, patch_center_y, patch_size)

        # check the patch dimensions
        self.assertEqual(computed_patch.width, patch_size)
        self.assertEqual(computed_patch.height, patch_size)

        # check for zeros in area out of bounds
        np.testing.assert_allclose(computed_patch.value_array[-1:, :], 0)
        np.testing.assert_allclose(computed_patch.value_array[:, -4:], 0)

        # check the patch contents with the original input
        np.testing.assert_allclose(
            computed_patch.value_array[:-1, :-4],
            input_image.value_array[
                patch_center_y - patch_size // 2 :,
                patch_center_x - patch_size // 2 :,
            ],
        )
