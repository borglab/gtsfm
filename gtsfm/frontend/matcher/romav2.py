"""RoMa v2 image matcher.

The network was proposed in "RoMa v2: Harder Better Faster Denser Feature Matching".

References:
- https://arxiv.org/abs/2511.15706
- https://github.com/Parskatt/RoMaV2

Install (not a pip extra: romav2 pins torchvision>=0.23, which conflicts with GTSfM):
    pip install romav2 --no-deps
"""

from typing import Optional, Tuple

import numpy as np
import PIL.Image
import torch

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase


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
            use_cuda: Use CUDA when a GPU is available. Defaults to True.
            min_confidence: Minimum overlap confidence required for matches. Defaults to 0.1.
            max_keypoints: Maximum number of sampled correspondences. Defaults to 8000.
            setting: RoMa v2 preset (`precise`, `base`, `fast`, `turbo`). Defaults to `precise`.
        """
        super().__init__()
        # Fail at construction when the package is missing. Do not build the network here:
        # RoMaV2() replaces DINOv3.forward on the class, and that replacement does not
        # travel with the matcher when Dask sends it to a worker process.
        try:
            import romav2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "RoMa v2 support requires the `romav2` package. Install with:\n"
                "  pip install romav2 --no-deps"
            ) from exc

        self._use_cuda = use_cuda
        self._min_confidence = min_confidence
        self._max_keypoints = max_keypoints
        self._setting = setting
        self._matcher = None
        self._device: Optional[torch.device] = None

    def _ensure_matcher(self) -> None:
        """Build RoMa in the process that is about to run match()."""
        if self._matcher is not None:
            return

        from romav2 import RoMaV2
        import romav2.device as romav2_device

        # v2 picks a process-global device at import time. Override it so use_cuda is honored.
        if self._use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        elif self._use_cuda:
            device = romav2_device.device
        else:
            device = torch.device("cpu")
        romav2_device.device = device
        torch.set_float32_matmul_precision("highest")

        matcher = RoMaV2()
        matcher.to(device=device)
        matcher.apply_setting(self._setting)
        matcher.eval()
        self._device = device
        self._matcher = matcher

    def match(self, image_i1: Image, image_i2: Image) -> Tuple[Keypoints, Keypoints]:
        """Identify feature matches across two images.

        Note: the matcher will run out of memory for large image sizes.

        Args:
            image_i1: first input image of pair.
            image_i2: second input image of pair.

        Returns:
            Keypoints from image 1 (N keypoints will exist).
            Corresponding keypoints from image 2 (there will also be N keypoints). These represent feature matches.
        """
        self._ensure_matcher()
        assert self._matcher is not None

        with torch.no_grad():
            im1 = PIL.Image.fromarray(image_i1.value_array).convert("RGB")
            im2 = PIL.Image.fromarray(image_i2.value_array).convert("RGB")
            preds = self._matcher.match(im1, im2)
            matches, overlaps = self._sample_correspondences(preds)

        matches = matches[overlaps > self._min_confidence]

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

    def _sample_correspondences(self, preds: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample matches. CPU cdist does not support the float16 path v2 uses by default."""
        assert self._device is not None and self._matcher is not None
        if self._device.type == "cuda":
            matches, overlaps, _, _ = self._matcher.sample(preds, self._max_keypoints)
            return matches, overlaps

        import romav2.romav2 as romav2_mod

        original_kde = romav2_mod.kde

        def _float32_kde(x, std: float = 0.1, half: bool = True):
            del half
            return original_kde(x, std=std, half=False)

        romav2_mod.kde = _float32_kde
        try:
            matches, overlaps, _, _ = self._matcher.sample(preds, self._max_keypoints)
        finally:
            romav2_mod.kde = original_kde
        return matches, overlaps
