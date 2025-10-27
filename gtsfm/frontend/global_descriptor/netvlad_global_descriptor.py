"""Wrapper around the NetVLAD global image descriptor.
Based on Arandjelovic16cvpr:
"NetVLAD: CNN architecture for weakly supervised place recognition"
https://arxiv.org/pdf/1511.07247.pdf
NetVLAD, is a new generalized VLAD layer, inspired by the “Vector of Locally Aggregated Descriptors”
image representation commonly used in image retrieval
Whereas bag-of-visual-words aggregation keeps counts of visual words, VLAD stores the sum of residuals
(difference vector between the descriptor and its corresponding cluster centre) for each visual word.
Authors: John Lambert, Travis Driver
"""

import numpy as np
import torch
from torchvision.transforms import v2 as transforms  # type: ignore

import gtsfm.utils.logger as logger_utils
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from gtsfm.loader.loader_base import BatchTransform, ResizeTransform
from thirdparty.hloc.netvlad import NetVLAD


class NetVLADGlobalDescriptor(GlobalDescriptorBase):
    """NetVLAD global descriptor"""

    def __init__(self) -> None:
        super().__init__()
        self._model: torch.nn.Module | None = None  # Lazy loading - only load when describe() is called

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the NetVLAD model to avoid unnecessary initialization."""
        if self._model is None:
            logger = logger_utils.get_logger()
            logger.info("⏳ Loading NetVLAD model weights...")
            self._model = NetVLAD().eval()

    def get_preprocessing_transforms(self) -> tuple[ResizeTransform, BatchTransform | None]:
        """Return per-image preprocessing and optional batch transforms."""
        resize_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: torch.from_numpy(np.array(x, copy=True))),
                transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # [H,W,C] → [C,H,W]
            ]
        )

        # Transform 2: Convert to float32 and normalize to [0, 1]
        batch_transform = transforms.Lambda(lambda x: x.type(torch.float32) / 255.0)

        return resize_transform, batch_transform

    def describe_batch(self, images: torch.Tensor) -> list[np.ndarray]:
        """Compute descriptors for a batch of images efficiently."""
        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        images = images.to(device)

        with torch.no_grad():
            # 2. Get all descriptors from the model in a single forward pass.
            batch_descriptors = self._model({"image": images})

        # 3. Convert the output tensor back to a list of numpy arrays.
        descriptors_np = batch_descriptors["global_descriptor"].detach().cpu().numpy()
        return [desc for desc in descriptors_np]
