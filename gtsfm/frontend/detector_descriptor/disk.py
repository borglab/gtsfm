"""DISK detector+descriptor implementation.

The network was proposed in 'DISK: Learning local features with policy gradient' and is implemented by wrapping over the 
authors' implementation.

Reference: https://github.com/cvlab-epfl/disk

Authors: Travis Driver
"""
import numpy as np
import torch

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase

from thirdparty.LightGlue.lightglue.disk import DISK


class DiskDetectorDescriptor(DetectorDescriptorBase):
    """DISK Detector+Descriptor implementation."""

    def __init__(
        self,
        max_keypoints: int = 8000,
        use_cuda: bool = True,
    ) -> None:
        """Configures the object.

        Args:
            max_keypoints: max keypoints to detect in an image.
            use_cuda (optional): flag controlling the use of GPUs via CUDA. Defaults to True.
            weights_path (optional): Path to the model weights. Defaults to MODEL_WEIGHT_PATH.
        """
        super().__init__(max_keypoints=max_keypoints)
        self._use_cuda = use_cuda
        # self._config = {"weights_path": weights_path}
        self._model = DISK(max_num_keypoints=max_keypoints).eval()

    def detect_and_describe(self, image: Image) -> Tuple[Keypoints, np.ndarray]:
        """Jointly generate keypoint detections and their associated descriptors from a single image."""
        device = torch.device("cuda" if self._use_cuda and torch.cuda.is_available() else "cpu")
        self._model.to(device)

        # Compute features.
        image_tensor = torch.from_numpy(
            np.expand_dims(image_utils.rgb_to_gray_cv(image).value_array.astype(np.float32) / 255.0, (0, 1))
        ).to(device)
        with torch.no_grad():
            model_results = self._model.extract(image_tensor, resize=None)
        torch.cuda.empty_cache()

        # Unpack results.
        coordinates = model_results["keypoints"][0].detach().cpu().numpy()
        scores = model_results["keypoint_scores"][0].detach().cpu().numpy()
        keypoints = Keypoints(coordinates, scales=None, responses=scores)
        descriptors = model_results["descriptors"][0].detach().cpu().numpy()

        # Filter features.
        if image.mask is not None:
            keypoints, valid_idxs = keypoints.filter_by_mask(image.mask)
            descriptors = descriptors[valid_idxs]
        keypoints, selection_idxs = keypoints.get_top_k(self.max_keypoints)
        descriptors = descriptors[selection_idxs]

        return keypoints, descriptors
