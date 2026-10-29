"""Regression tests for RootSIFT computed from OpenCV's original SIFT frames."""

import pickle
import unittest
from unittest.mock import patch

import cv2 as cv
import numpy as np

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.descriptor.sift import SIFTDescriptor
from gtsfm.frontend.detector_descriptor.rootsift import RootSIFTDetectorDescriptor
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor


class TestRootSIFT(unittest.TestCase):
    """Check preservation of SIFT frames, normalization, and matcher compatibility."""

    def setUp(self) -> None:
        """Create a deterministic textured image without loading a benchmark dataset."""
        pixels = np.random.default_rng(0).integers(0, 256, (192, 192, 3), dtype=np.uint8)
        self.image = Image(value_array=pixels)
        self.extractor = RootSIFTDetectorDescriptor(max_keypoints=20)

    def test_preserves_original_descriptors(self) -> None:
        """Normalize original SIFT descriptors, avoiding the lossy keypoint round trip."""
        sift_keypoints, sift = SIFTDetectorDescriptor(max_keypoints=20).detect_and_describe(self.image)
        keypoints, rootsift = self.extractor.detect_and_describe(self.image)
        self.assertEqual(len(keypoints), 20)
        self.assertEqual(keypoints, sift_keypoints)
        # Match the normalization used in the recorded benchmark adapter.
        expected = np.sqrt(sift / (sift.sum(axis=1, keepdims=True) + np.finfo(float).eps)).astype(np.float32)
        np.testing.assert_array_equal(rootsift, expected)
        np.testing.assert_allclose(np.linalg.norm(rootsift, axis=1), 1, atol=1e-6)
        self.assertEqual(rootsift.dtype, np.float32)

        # The old standalone path recomputed descriptors with missing frame metadata.
        recomputed = SIFTDescriptor().describe(self.image, keypoints)
        self.assertFalse(np.allclose(sift, recomputed))

    def test_opencv_matching(self) -> None:
        """Float32 RootSIFT descriptors can be passed directly to OpenCV's matcher."""
        _, descriptors = self.extractor.detect_and_describe(self.image)
        matches = cv.BFMatcher(cv.NORM_L2, crossCheck=True).match(descriptors, descriptors)
        self.assertEqual(len(matches), len(descriptors))
        self.assertTrue(all(match.distance == 0 for match in matches))

    def test_mask(self) -> None:
        """Restrict detections to the requested part of the image."""
        mask = np.zeros((192, 192), dtype=np.uint8)
        mask[:, :96] = 1
        image = Image(value_array=self.image.value_array, mask=mask)
        keypoints, descriptors = self.extractor.detect_and_describe(image)
        self.assertGreater(len(keypoints), 0)
        self.assertTrue(np.all(keypoints.coordinates[:, 0] < 96))
        self.assertEqual(descriptors.shape, (len(keypoints), 128))

    def test_no_detections(self) -> None:
        """Blank images and fully masked images yield well-shaped empty outputs."""
        blank = Image(value_array=np.zeros((64, 64, 3), dtype=np.uint8))
        masked = Image(value_array=self.image.value_array, mask=np.zeros((192, 192), dtype=np.uint8))
        for extractor in (SIFTDetectorDescriptor(), self.extractor):
            for image in (blank, masked):
                with self.subTest(extractor=type(extractor).__name__, masked=image.mask is not None):
                    keypoints, descriptors = extractor.detect_and_describe(image)
                    self.assertEqual(keypoints.coordinates.shape, (0, 2))
                    self.assertEqual(descriptors.shape, (0, 128))
                    self.assertEqual(descriptors.dtype, np.float32)

    def test_zero_descriptor(self) -> None:
        """A zero histogram stays finite and zero after normalization."""
        keypoints = Keypoints(coordinates=np.array([[10, 10]]))
        descriptors = np.zeros((1, 128), dtype=np.float32)
        with patch.object(SIFTDetectorDescriptor, "detect_and_describe", return_value=(keypoints, descriptors)):
            _, rootsift = self.extractor.detect_and_describe(self.image)
        np.testing.assert_array_equal(rootsift, descriptors)
        self.assertEqual(rootsift.dtype, np.float32)

    def test_pickleable(self) -> None:
        """Dask serialization preserves the extractor and feature limit."""
        restored = pickle.loads(pickle.dumps(self.extractor))
        self.assertIsInstance(restored, RootSIFTDetectorDescriptor)
        self.assertEqual(restored.max_keypoints, 20)
