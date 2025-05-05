"""VLFeat SIFT Detector-Descriptor implementation.

The detector was proposed in 'Distinctive Image Features from Scale-Invariant Keypoints' and is implemented by wrapping
over PyVLFeat's API, a Python wrapper of the VLFeat library.

References:
- https://github.com/vlfeat/vlfeat
- https://github.com/u1234x1234/pyvlfeat/tree/master
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Matthew Woodward
"""
from typing import Tuple

import vlfeat
import numpy as np
import copy

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase


class VLFeatDetectorDescriptor(DetectorDescriptorBase):
    """SIFT detector-descriptor using VLFeat's implementation."""

    def __init__(self, max_keypoints: int = 5000, use_upright: bool = True, use_root: bool = True):
        """Initialize the detector-descriptor.

        Args:
            max_keypoints: Maximum number of keypoints to detect. Defaults to 5000.
            use_upright: Whether to use SIFT in upright mode, which uses the first orientation detected at a keypoint
                         location. Defaults to True.
            use_root: Whether to use root SIFT descriptors, which normalizes the 128-element descriptor by its L1-norm.
                         Defaults to True.
        """
        super(VLFeatDetectorDescriptor, self).__init__(max_keypoints)

        self.use_upright = use_upright
        self.use_root = use_root

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Perform feature detection as well as their description.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for details about the output format.

        Args:
            image: The input image.

        Returns:
            Detected keypoints, with length N <= max_keypoints.
            Corr. descriptors, of shape (N, D) where D is the dimension of each descriptor.
        """

        def get_valid_first_octave(first_octave: int, height: int, width: int) -> int:
            """Determine whether the requested first octave for the SIFT Gaussian pyramid is valid and calculate a new
            one if needed.

            Args:
                first_octave: The index of the requested first octave.
                width: Width of input image.
                height: Height of input image.

            Returns:
                valid_first_octave: The index of the first valid octave.
            """
            kMaxScaledDim = 3600
            max_dim = max(width, height)
            valid_first_octave = first_octave
            scale_factor = 2.0 ** (-1 * valid_first_octave)
            while (max_dim * scale_factor) >= kMaxScaledDim:
                scale_factor /= 2.0
                valid_first_octave += 1

            return valid_first_octave

        def convert_to_root_sift(descriptor: np.ndarray) -> np.ndarray:
            """Convert a given descriptor to its root form by normalizing by its L1-norm and taking the square root
            over all elements.

            Args:
                descriptor: Descriptor to convert to root form.

            Returns:
                root_desc: Descriptor converted to root form.
            """
            root_desc = copy.deepcopy(descriptor)
            kTolerance = 1e-8
            l1_norm = np.linalg.norm(root_desc, ord=1)
            if l1_norm > kTolerance:
                root_desc /= l1_norm
                root_desc = np.sqrt(root_desc)

            return root_desc

        # Establish parameters.
        opt_frames = np.array([])
        opt_octaves = -1
        opt_levels = 3
        opt_first_octave = -1
        opt_peak_thresh = 1.2
        opt_edge_thresh = 10
        opt_upright_sift = self.use_upright
        opt_verbose = 0

        # Get valid first octave.
        h, w, _ = image.shape
        valid_first_octave = get_valid_first_octave(opt_first_octave, height=h, width=w)

        # Scale image and parameters to range [0, 1] to match Theia's usage of VLFeat.
        scaled_image = Image(value_array=image.value_array.astype(np.float32) / 255.0,
                             exif_data=image.exif_data,
                             file_name=image.file_name)
        opt_peak_thresh /= 255.0
        opt_edge_thresh /= 255.0

        # Convert to grayscale.
        gray_image = image_utils.rgb_to_gray_cv(scaled_image)

        # Run the VLFeat code.
        vl_keypoints, descriptors = vlfeat.vl_sift(gray_image.value_array,
                                                   opt_frames,
                                                   opt_octaves,
                                                   opt_levels,
                                                   valid_first_octave,
                                                   opt_peak_thresh,
                                                   opt_edge_thresh,
                                                   upright_sift=opt_upright_sift,
                                                   verbose=opt_verbose)

        # Re-orient vl_keypoints and descriptors because VLFeat uses column-major matrix convention.
        vl_keypoints, descriptors = vl_keypoints.T, descriptors.T

        # Save only up to the max number of keypoints allowed.
        if vl_keypoints.shape[0] > self.max_keypoints:
            vl_keypoints = vl_keypoints[:self.max_keypoints]
            descriptors = descriptors[:self.max_keypoints]

        # Convert descriptors to root form if desired.
        if self.use_root:
            root_descriptors = np.zeros_like(descriptors)
            for i in range(descriptors.shape[0]):
                root_desc = convert_to_root_sift(descriptors[i])
                root_descriptors[i] = root_desc
            descriptors = copy.deepcopy(root_descriptors)

        # Convert to GTSFM's keypoints.
        keypoints = Keypoints(coordinates=vl_keypoints[:, 0:2], scales=vl_keypoints[:, 2].reshape(-1, 1))

        return keypoints, descriptors