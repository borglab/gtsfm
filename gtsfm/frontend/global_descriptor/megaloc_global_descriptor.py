""" Wrapper around the MegaLoc Global Descriptor
    Based on Gabrielle Berton's
    "MegaLoc: One Retrieval to Place Them All"
    https://arxiv.org/pdf/2502.17237

    Authors: Kathir Gounder
"""

from contextlib import contextmanager
import threading
from typing import Optional

import numpy as np
import torch
from dask.distributed import Lock as DaskLock, get_client

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase


class MegaLocGlobalDescriptor(GlobalDescriptorBase):
    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[torch.nn.Module] = None
        self._local_lock = threading.Lock()

    @contextmanager
    def _model_load_lock(self):
        """Coordinate model downloads across distributed workers."""
        try:
            # Use a distributed lock when a scheduler is available (typical during Dask execution).
            get_client()  # raises ValueError if no active client
            lock = DaskLock("gtsfm_megaloc_model_load")
            with lock:
                yield
        except Exception:
            # Fallback to a process-local lock (e.g., when running without Dask).
            with self._local_lock:
                yield

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the MegaLoc Model to avoid unnecessary initialization."""
        from thirdparty.megaloc.megaloc import MegaLocModel

        if self._model is not None:
            return

        with self._model_load_lock():
            if self._model is None:
                logger = logger_utils.get_logger()
                logger.info("⏳ Loading MegaLoc model weights...")
                self._model = MegaLocModel().eval()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_local_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local_lock = threading.Lock()

    def describe(self, image: Image) -> np.ndarray:
        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        img_array = image.value_array.copy()
        img_tensor = torch.from_numpy(img_array).to(device).permute(2, 0, 1).unsqueeze(0).type(torch.float32) / 255.0

        with torch.no_grad():
            descriptor = self._model(img_tensor)

        return descriptor.detach().squeeze().cpu().numpy()
