""" Wrapper around the MegaLoc Global Descriptor
    Based on Gabrielle Berton's 
    "MegaLoc: One Retrieval to Place Them All"
    https://arxiv.org/pdf/2502.17237

    Authors: Kathir Gounder
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase


class MegaLocGlobalDescriptor(GlobalDescriptorBase):
    def __init__(self, input_size: int = 322) -> None:
        super().__init__()
        self._model: Optional[torch.nn.Module] = None
        self._input_size = input_size
        
    def _ensure_model_loaded(self) -> None:
        """Lazy-load the MegaLoc Model to avoid unnecessary initialization"""
        from thirdparty.megaloc.megaloc import MegaLocModel

        if self._model is None:
            logger = logger_utils.get_logger()
            logger.info("⏳ Loading MegaLoc model weights...")
            self._model = MegaLocModel().eval()

    def _resize_image_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tensor: Input tensor of shape [B, C, H, W]
        
        Returns:
            Resized tensor of shape [B, C, 322, 322]
        """
        _, _, h, w = img_tensor.shape
            
        img_tensor = F.interpolate(
            img_tensor, 
            size=(self._input_size, self._input_size),  # Square resize
            mode='bilinear', 
            align_corners=False,
            antialias=True
        )
    
        return img_tensor

    def describe(self, image: Image) -> np.ndarray:
        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        
        img_array = image.value_array.copy()
        img_tensor = (
            torch.from_numpy(img_array)
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .type(torch.float32) / 255.0
        )
        
        img_tensor = self._resize_image_tensor(img_tensor)

        with torch.no_grad():
            descriptor = self._model(img_tensor)
            
        return descriptor.detach().squeeze().cpu().numpy()

