"""RoMa v2 image matcher.

The network was proposed in "RoMa v2: Harder Better Faster Denser Feature Matching".

References:
- https://arxiv.org/abs/2511.15706
- https://github.com/Parskatt/RoMaV2

Authors: Kusum
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import PIL.Image
import torch

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


class RoMaV2Matcher(ImageMatcherBase):
    """RoMa v2 dense image matcher."""

    def __init__(
        self,
        use_cuda: bool = True,
        min_confidence: float = 0.1,
        max_keypoints: int = 8000,
        setting: str = "precise",
    ) -> None:
        """Initialize the matcher.

        Args:
            use_cuda: If True, prefer CUDA when available (otherwise romav2's MPS/CPU fallback).
                If False, force CPU even on a GPU host. Defaults to True.
            min_confidence: Minimum overlap confidence required for matches. Defaults to 0.1.
            max_keypoints: Maximum number of sampled correspondences. Defaults to 8000.
            setting: RoMa v2 preset (`precise`, `base`, `fast`, `turbo`, or benchmark names).
                Defaults to `precise`.
        """
        super().__init__()
        try:
            from romav2 import RoMaV2
        except ImportError as exc:
            raise ImportError(
                "RoMa v2 support requires the `romav2` package. Install with:\n"
                "  pip install romav2 --no-deps\n"
                "(Use --no-deps because romav2 declares torchvision>=0.23, which conflicts "
                "with GTSfM's torch<2.8 / torchvision<0.23 pin; the matcher runs on the "
                "pinned stack.)"
            ) from exc

        self._min_confidence = min_confidence
        self._max_keypoints = max_keypoints
        self._setting = setting
        self._use_cuda = use_cuda

        # RoMaV2.forward requires highest float32 matmul precision.
        torch.set_float32_matmul_precision("highest")

        # romav2 v2.0.1 picks a process-global device at import time from hardware
        # availability alone. Override it before constructing RoMaV2 so use_cuda is honored
        # (weights load + image tensors both follow romav2.device.device).
        import romav2.device as romav2_device

        if use_cuda and torch.cuda.is_available():
            target_device = torch.device("cuda")
        elif use_cuda:
            # Keep romav2's non-CUDA fallback (MPS/CPU) but warn that CUDA was requested.
            target_device = romav2_device.device
            logger.warning(
                "RoMa v2 requested CUDA but no GPU is available; using device=%s.",
                target_device,
            )
        else:
            target_device = torch.device("cpu")
        romav2_device.device = target_device

        logger.info("⏳ Loading RoMa v2 model weights (setting=%s, device=%s)...", setting, target_device)
        self._matcher = RoMaV2()
        self._matcher.to(target_device)
        self._matcher.apply_setting(setting)
        self._matcher.eval()
        self._device = target_device

    def match(self, image_i1: Image, image_i2: Image) -> Tuple[Keypoints, Keypoints]:
        """Identify feature matches across two images.

        Note: the matcher can use substantial GPU memory for large images / precise settings.

        Args:
            image_i1: first input image of pair.
            image_i2: second input image of pair.

        Returns:
            Keypoints from image 1 (N keypoints will exist).
            Corresponding keypoints from image 2 (there will also be N keypoints). These represent feature matches.
        """
        im1 = PIL.Image.fromarray(image_i1.value_array).convert("RGB")
        im2 = PIL.Image.fromarray(image_i2.value_array).convert("RGB")

        with torch.inference_mode():
            preds = self._matcher.match(im1, im2)
            matches, overlaps, _, _ = self._sample_correspondences(preds)

        keep = overlaps > self._min_confidence
        matches = matches[keep]

        h1, w1 = image_i1.shape[:2]
        h2, w2 = image_i2.shape[:2]
        mkpts1, mkpts2 = self._matcher.to_pixel_coordinates(matches, h1, w1, h2, w2)

        keypoints_i1 = Keypoints(coordinates=mkpts1.detach().cpu().numpy())
        keypoints_i2 = Keypoints(coordinates=mkpts2.detach().cpu().numpy())

        valid_ind = np.arange(len(keypoints_i1))
        if image_i1.mask is not None:
            _, valid_ind_i1 = keypoints_i1.filter_by_mask(image_i1.mask)
            valid_ind = np.intersect1d(valid_ind, valid_ind_i1)
        if image_i2.mask is not None:
            _, valid_ind_i2 = keypoints_i2.filter_by_mask(image_i2.mask)
            valid_ind = np.intersect1d(valid_ind, valid_ind_i2)

        return keypoints_i1.extract_indices(valid_ind), keypoints_i2.extract_indices(valid_ind)

    def _sample_correspondences(self, preds: dict):
        """Sample matches from dense RoMa v2 predictions.

        romav2 v2.0.1's ``sample()`` calls ``kde(..., half=True)``, which runs
        ``torch.cdist`` on float16 tensors. PyTorch's CPU (and often MPS) cdist
        kernels do not support float16, so we temporarily force ``half=False`` on
        non-CUDA devices.
        """
        import romav2.romav2 as romav2_mod

        if self._device.type == "cuda":
            return self._matcher.sample(preds, self._max_keypoints)

        original_kde = romav2_mod.kde

        def _float32_kde(x, std: float = 0.1, half: bool = True):
            del half  # ignore romav2's default; CPU/MPS need float32 cdist
            return original_kde(x, std=std, half=False)

        romav2_mod.kde = _float32_kde
        try:
            return self._matcher.sample(preds, self._max_keypoints)
        finally:
            romav2_mod.kde = original_kde

