"""SIFT Detector-Descriptor implementation.

The detector was proposed in 'Distinctive Image Features from Scale-Invariant Keypoints' and is implemented by wrapping
over OpenCV's API.

References:
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np
import pycolmap

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


# Cached after the first call. "legacy" is Sift(SiftExtractionOptions), used by pycolmap before 3.13.
# "feature" is Sift(FeatureExtractionOptions), and "feature_cpu" is that constructor forced onto CPU
# when the default device cannot create a GPU extractor.
_SIFT_CTOR: str | None = None


def _cpu_sift(feature_options: "pycolmap.FeatureExtractionOptions") -> pycolmap.Sift:
    """Construct the post-3.13 extractor on CPU."""
    feature_options.use_gpu = False
    device = getattr(getattr(pycolmap, "Device", None), "cpu", None)
    if device is None:
        return pycolmap.Sift(feature_options)
    return pycolmap.Sift(feature_options, device=device)


def _colmap_sift(max_num_features: int) -> pycolmap.Sift:
    """Build a SIFT extractor for the installed pycolmap constructor."""
    global _SIFT_CTOR
    sift_options = pycolmap.SiftExtractionOptions(max_num_features=max_num_features)
    if _SIFT_CTOR is None:
        try:
            extractor = pycolmap.Sift(sift_options)
        except TypeError:
            _SIFT_CTOR = "feature"
        else:
            _SIFT_CTOR = "legacy"
            return extractor

    if _SIFT_CTOR == "legacy":
        return pycolmap.Sift(sift_options)

    feature_options = pycolmap.FeatureExtractionOptions(sift=sift_options)
    if _SIFT_CTOR == "feature_cpu":
        return _cpu_sift(feature_options)
    try:
        return pycolmap.Sift(feature_options)
    except (RuntimeError, ValueError):
        _SIFT_CTOR = "feature_cpu"
        return _cpu_sift(feature_options)


class ColmapSIFTDetectorDescriptor(DetectorDescriptorBase):
    """SIFT detector-descriptor using Colmap's implementation."""

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for details about the output format.

        Args:
            image: the input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """

        # Convert to grayscale.
        gray_image = image_utils.rgb_to_gray_cv(image)

        # Create pycolmap object every time as the object is not pickle-able.
        # pycolmap < 3.13 takes SiftExtractionOptions. Later versions take FeatureExtractionOptions
        # and default to the GPU extractor, which falls back to CPU when that extractor cannot be created.
        sift_obj = _colmap_sift(self.max_keypoints)

        # Extract features. pycolmap >= 0.5 returns (features, descriptors); earlier builds also return scores.
        extracted = sift_obj.extract(gray_image.value_array.astype(np.float32))
        features, descriptors = extracted[0], extracted[-1]

        # Convert to GTSFM's keypoints.
        # Note: Columns of features is x-coordinate, y-coordinate, scale, and orientation, respectively.
        keypoints = Keypoints(coordinates=features[..., :2], scales=features[:, 2])

        return keypoints, descriptors
