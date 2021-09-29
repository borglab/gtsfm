"""Superpoint detector+descriptor implementation.

The network was proposed in 'SuperPoint: Self-Supervised Interest Point Detection and Description' and is implemented
by wrapping over the authors' implementation.

References:
- https://arxiv.org/abs/1712.07629
- https://github.com/magicleap/SuperGluePretrainedNetwork

Authors: Ayush Baid
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import dask
import numpy as np
import torch
from dask.delayed import Delayed

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase
from thirdparty.SuperGluePretrainedNetwork.models.superpoint import SuperPoint

ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent
MODEL_WEIGHTS_PATH = (
    ROOT_PATH / "thirdparty" / "SuperGluePretrainedNetwork" / "models" / "weights" / "superpoint_v1.pth"
)


class SuperPointDetectorDescriptor(DetectorDescriptorBase):
    """Superpoint Detector+Descriptor implementation."""

    def __init__(self, use_cuda: bool = True, weights_path: Union[Path, str] = MODEL_WEIGHTS_PATH) -> None:
        """Configures the object.

        Args:
            max_keypoints: max keypoints to detect in an image.
            use_cuda (optional): flag controlling the use of GPUs via CUDA. Defaults to True.
            weights_path (optional): Path to the model weights. Defaults to MODEL_WEIGHT_PATH.
        """
        super().__init__()
        self._config = {"weights_path": weights_path}
        self._extra_args = None

    def detect_and_describe(
        self, image: Image, extra_args: Optional[Dict[Any, Any]] = None
    ) -> Tuple[Keypoints, np.ndarray]:
        """Jointly generate keypoint detections and their associated descriptors from a single image."""
        model = extra_args["model"]
        # TODO: fix inference issue #110

        image_tensor = torch.from_numpy(
            np.expand_dims(image_utils.rgb_to_gray_cv(image).value_array.astype(np.float32) / 255.0, (0, 1))
        ).to("cpu")

        with torch.no_grad():
            model_results = model({"image": image_tensor})

        torch.cuda.empty_cache()

        feature_points = model_results["keypoints"][0].detach().cpu().numpy()
        scores = model_results["scores"][0].detach().cpu().numpy()
        descriptors = model_results["descriptors"][0].detach().cpu().numpy().T

        # sort by scores
        sort_idxs = np.argsort(-scores)
        # limit the number of keypoints
        sort_idxs = sort_idxs[: self.max_keypoints]
        feature_points = feature_points[sort_idxs]
        scores = scores[sort_idxs]
        descriptors = descriptors[sort_idxs]

        keypoints = Keypoints(feature_points, responses=scores, scales=np.ones(scores.shape))

        return keypoints, descriptors

    def get_delayed_extra_args(self) -> Delayed:
        if self._extra_args is None:
            model = SuperPoint(self._config).cpu()
            extra_args = {"model": model}
            self._extra_args = dask.delayed(extra_args)

        return self._extra_args

    def create_computation_graph(self, image_graph: Delayed) -> Tuple[Delayed, Delayed]:
        return dask.delayed(self.detect_and_describe, nout=2)(image_graph, self.get_delayed_extra_args())
