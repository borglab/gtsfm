"""RoMa v2 image matcher.

The network was proposed in "RoMa v2: Harder Better Faster Denser Feature Matching".

References:
- https://arxiv.org/abs/2511.15706
- https://github.com/Parskatt/RoMaV2
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
import PIL.Image
import torch

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

# romav2 monkey-patches DinoVisionTransformer.forward on the class; that patch does not
# survive cross-process pickle (Dask workers). Rebuild per worker and cache the weights.
_WORKER_MATCHER_CACHE: Dict[Tuple[Any, ...], Tuple[Any, torch.device]] = {}
_WORKER_MATCHER_LOCK = threading.Lock()
# File lock: Dask may run nested tasks concurrently even with threads_per_worker=1;
# a threading.Lock is not enough to keep CUDA forwards serial.
_GPU_LOCK_PATH = os.environ.get("GTSFM_ROMAV2_GPU_LOCK", "/tmp/gtsfm_romav2_gpu.lock")


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
            import romav2  # noqa: F401
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
        self._matcher: Optional[Any] = None
        self._device: Optional[torch.device] = None
        # Lazy-load on first match() so Dask can scatter a lightweight config object.

    def __getstate__(self) -> Dict[str, Any]:
        """Omit weights from the pickle payload; workers rebuild via ``_ensure_matcher``."""
        state = self.__dict__.copy()
        state["_matcher"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._matcher = None
        # Do not load weights during unpickle; first match() will call _ensure_matcher.

    def _cache_key(self) -> Tuple[Any, ...]:
        return (self._setting, self._use_cuda, torch.cuda.is_available())

    def _ensure_matcher(self) -> None:
        """Load RoMa v2 weights, reusing a per-process cache after Dask unpickle."""
        if self._matcher is not None:
            return

        with _WORKER_MATCHER_LOCK:
            if self._matcher is not None:
                return

            cache_key = self._cache_key()
            cached = _WORKER_MATCHER_CACHE.get(cache_key)
            if cached is not None:
                self._matcher, self._device = cached
                self._prepare_process_state(self._device)
                return

            from romav2 import RoMaV2
            import romav2.device as romav2_device

            if self._use_cuda and torch.cuda.is_available():
                target_device = torch.device("cuda")
            elif self._use_cuda:
                target_device = romav2_device.device
                logger.warning(
                    "RoMa v2 requested CUDA but no GPU is available; using device=%s.",
                    target_device,
                )
            else:
                target_device = torch.device("cpu")

            self._prepare_process_state(target_device)

            logger.info(
                "⏳ Loading RoMa v2 model weights (setting=%s, device=%s)...",
                self._setting,
                target_device,
            )
            matcher = RoMaV2()
            matcher.to(device=target_device)
            matcher.apply_setting(self._setting)
            matcher.eval()

            self._matcher = matcher
            self._device = target_device
            _WORKER_MATCHER_CACHE[cache_key] = (matcher, target_device)

    def _prepare_process_state(self, device: torch.device) -> None:
        """Re-apply process-local romav2 globals (lost across Dask worker processes)."""
        import os

        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        # Avoid torch._inductor compile-worker storms under Dask (can hang first match).
        import torch._dynamo

        torch._dynamo.config.disable = True
        try:
            torch.compiler.set_stance("force_eager")
        except Exception:
            pass
        torch.set_float32_matmul_precision("highest")
        import romav2.device as romav2_device

        romav2_device.device = device

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
        self._ensure_matcher()
        assert self._matcher is not None and self._device is not None
        self._prepare_process_state(self._device)

        im1 = PIL.Image.fromarray(image_i1.value_array).convert("RGB")
        im2 = PIL.Image.fromarray(image_i2.value_array).convert("RGB")

        import fcntl

        with open(_GPU_LOCK_PATH, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
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

        assert self._matcher is not None and self._device is not None

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
