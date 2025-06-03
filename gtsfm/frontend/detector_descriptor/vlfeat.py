"""VLFeat SIFT Detector-Descriptor implementation.

The detector was proposed in 'Distinctive Image Features from Scale-Invariant Keypoints' and is implemented by wrapping
over PyVLFeat's API, a Python wrapper of the VLFeat library.

References:
- https://github.com/vlfeat/vlfeat
- https://github.com/u1234x1234/pyvlfeat/tree/master
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Authors: Matthew Woodward
"""
import copy
from typing import Tuple

import numpy as np
import vlfeat

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase


class VLFeatDetectorDescriptor(DetectorDescriptorBase):
    """SIFT detector-descriptor using VLFeat's implementation."""

    def __init__(
        self,
        max_keypoints: int = 5000,
        max_scaled_dim: int = 3600,
        num_octaves: int = -1,
        num_levels: int = 3,
        first_octave: int = -1,
        peak_thresh: float = 1.2,
        edge_thresh: float = 10,
        use_upright: bool = False,
        use_root: bool = True
    ) -> None:
        """Initialize the detector-descriptor.

        Args:
            max_keypoints: Maximum number of keypoints to detect. Defaults to 5000.
            max_scaled_dim: Maximum size input image is scaled to in height or width. Defaults to 3600 pixels.
            num_octaves: Number of octaves in the DoG scale space. Defaults to -1, which uses the max available octaves.
            num_levels: Number of levels in the DoG scale space. Defaults to 3.
            first_octave: Index of the first octave in the DoG scale space. Negative indices indicate an upsampling of
                          both image dimensions by a corresponding factor of 2. Defaults to -1.
            peak_thresh: Peak selection threshold in the absolute DoG scale space. Defaults to 1.2.
            edge_thresh: Non-edge selection threshold. Defaults to 10.
            use_upright: Whether to use SIFT in upright mode, which uses the first orientation detected at a keypoint
                         location. Defaults to True.
            use_root: Whether to use root SIFT descriptors, which normalizes the 128-element descriptor by its L1-norm.
                      Defaults to True.
        """
        super(VLFeatDetectorDescriptor, self).__init__(max_keypoints)

        self.max_scaled_dim = max_scaled_dim
        self.num_octaves = num_octaves
        self.num_levels = num_levels
        self.first_octave = first_octave
        self.peak_thresh = peak_thresh
        self.edge_thresh = edge_thresh
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
            max_dim = max(width, height)
            valid_first_octave = first_octave
            scale_factor = 2.0 ** (-1 * valid_first_octave)

            # Reduce the scale factor by 2.0 while the first octave puts the max image dimension above limit.
            while (max_dim * scale_factor) >= self.max_scaled_dim:
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
            kTolerance = 1e-8

            root_desc = copy.deepcopy(descriptor)
            l1_norm = np.linalg.norm(root_desc, ord=1)

            # Normalize the descriptor by its L1-norm and compute square root.
            if l1_norm > kTolerance:
                root_desc /= l1_norm
                root_desc = np.sqrt(root_desc)

            return root_desc

        # Establish parameters.
        opt_num_octaves = self.num_octaves
        opt_num_levels = self.num_levels
        opt_first_octave = self.first_octave
        opt_peak_thresh = self.peak_thresh
        opt_edge_thresh = self.edge_thresh
        opt_upright_sift = self.use_upright

        # Get valid first octave.
        h, w, _ = image.shape
        valid_first_octave = get_valid_first_octave(opt_first_octave, height=h, width=w)

        # Check if image is between [0, 255] and rescale to [0, 1].
        if np.max(image.value_array) > 1:
            # Scale image and parameters to range [0, 1] to match Theia's usage of VLFeat.
            image = Image(
                value_array=image.value_array.astype(np.float32) / 255.0,
                exif_data=image.exif_data,
                file_name=image.file_name
            )
            opt_peak_thresh /= 255.0
            opt_edge_thresh /= 255.0

        # Convert to grayscale.
        gray_image = image_utils.rgb_to_gray_cv(image)

        # Run the VLFeat code.
        vl_keypoints, descriptors = vlfeat.vl_sift(
            gray_image.value_array,
            octaves=opt_num_octaves,
            levels=opt_num_levels,
            first_octave=valid_first_octave,
            peak_thresh=opt_peak_thresh,
            edge_thresh=opt_edge_thresh,
            upright_sift=opt_upright_sift
        )

        # Re-orient vl_keypoints and descriptors because VLFeat uses column-major matrix convention.
        vl_keypoints, descriptors = vl_keypoints.T, descriptors.T

        # Convert descriptors to root form if desired.
        if self.use_root:
            root_descriptors = np.zeros_like(descriptors)
            for i in range(descriptors.shape[0]):
                root_desc = convert_to_root_sift(descriptors[i])
                root_descriptors[i] = root_desc
            descriptors = copy.deepcopy(root_descriptors)

        # Convert to GTSFM's keypoints.
        keypoints = Keypoints(coordinates=vl_keypoints[:, 0:2], scales=vl_keypoints[:, 2])

        # Filter features.
        keypoints, selection_idxs = keypoints.get_top_k(self.max_keypoints)
        descriptors = descriptors[selection_idxs]

        return keypoints, descriptors
