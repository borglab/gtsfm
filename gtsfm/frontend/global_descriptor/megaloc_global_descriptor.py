""" Wrapper around the MegaLoc Global Descriptor
    Based on Gabrielle Berton's 
    "MegaLoc: One Retrieval to Place Them All"
    https://arxiv.org/pdf/2502.17237

    Authors: Kathir Gounder
"""
import numpy as np
import torch
from typing import List, Optional

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase


class MegaLocGlobalDescriptor(GlobalDescriptorBase):
    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[torch.nn.Module] = None
        
    def _ensure_model_loaded(self) -> None:
        """Lazy-load the MegaLoc Model to avoid unnecessary initialization"""
        from thirdparty.megaloc.megaloc import MegaLocModel

        if self._model is None:
            logger = logger_utils.get_logger()
            logger.info("⏳ Loading MegaLoc model weights...")
            self._model = MegaLocModel().eval()

    def desecribe_batch(self, images: List[Image]) -> List[np.ndarray]:
        """Process multiple images in a single forward pass.
        
        Args:
            images: List of images to process
            
        Returns:
            descriptors: Array of shape (N, D) where N is number of images
        """
        self._ensure_model_loaded()
        assert self._model is not None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        
        # Batch preprocessing
        batch_tensors = []
        for image in images:
            img_array = image.value_array.copy()
            img_tensor = (
                torch.from_numpy(img_array)
                .permute(2, 0, 1)
                .type(torch.float32) / 255.0
            )
            batch_tensors.append(img_tensor)
        
        # Stack into batch [B, C, H, W]
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            descriptors = self._model(batch)
        
        # Need to unpack into a List of numpy arrays
        return [desc.detach().squeeze().cpu().numpy() for desc in descriptors]

    def describe(self, image: Image) -> np.ndarray:
        return self.describe_batch([image])[0]
