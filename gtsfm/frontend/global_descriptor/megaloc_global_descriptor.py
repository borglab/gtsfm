"""Wrapper around the MegaLoc Global Descriptor
Based on Gabrielle Berton
"MegaLoc: One Retrieval to Place Them All"
https://arxiv.org/pdf/2502.17237

Authors: Kathir Gounder
"""

import numpy as np
import torch
from torchvision.transforms import v2 as transforms  # type: ignore

import gtsfm.utils.logger as logger_utils
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from gtsfm.loader.loader_base import BatchTransform, ResizeTransform


class MegaLocGlobalDescriptor(GlobalDescriptorBase):
    def __init__(self) -> None:
        super().__init__()
        self._model: torch.nn.Module | None = None

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the MegaLoc Model to avoid unnecessary initialization"""
        from thirdparty.megaloc.megaloc import MegaLocModel

        if self._model is None:
            logger = logger_utils.get_logger()
            logger.info("⏳ Loading MegaLoc model weights...")
            self._model = MegaLocModel().eval()

    def get_preprocessing_transforms(self) -> tuple[ResizeTransform, BatchTransform | None]:
        """Return transforms for Megaloc preprocessing.

        Resize transform: permute + resize (applied individually)
        Batch transform: dtype conversion + normalization (applied to batch)
        """
        # Transform 1: Permute [H,W,C]→[C,H,W] and resize
        resize_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: torch.from_numpy(x)),
                transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # [H,W,C] → [C,H,W]
                transforms.Resize(size=(322, 322), antialias=True),  # Expects [C,H,W]
            ]
        )

        # Transform 2: Convert to float32 and normalize to [0, 1]
        batch_transform = transforms.Lambda(lambda x: x.type(torch.float32) / 255.0)

        return resize_transform, batch_transform

    def describe_batch(self, images: torch.Tensor) -> list[np.ndarray]:
        """Process multiple images in a single forward pass.

        Args:
            images: list of images to process

        Returns:
            descriptors: Array of shape (N, D) where N is number of images
        """
        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        images = images.to(device)

        with torch.no_grad():
            descriptors = self._model(images)

        # Need to unpack into a list of numpy arrays
        return [desc.detach().squeeze().cpu().numpy() for desc in descriptors]
