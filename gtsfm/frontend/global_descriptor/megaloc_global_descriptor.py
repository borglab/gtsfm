""" Wrapper around the MegaLoc Global Descriptor
    Based on Gabrielle Berton's 
    "MegaLoc: One Retrieval to Place Them All"
    https://arxiv.org/pdf/2502.17237

    Authors: Kathir Gounder
"""
import numpy as np
import torch
from torchvision import transforms
from typing import List, Optional, Callable

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
    
    def get_preprocessing_transform(self) -> Optional[Callable]:
        """Return transform to resize images to 322x322 square.
        
        This follows the MegaLoc paper's inference protocol (Section 3.1).
        """
        return transforms.Resize(size=(322, 322), antialias=True)
        
    def describe_batch(self, images: torch.Tensor) -> List[np.ndarray]:
        """Process multiple images in a single forward pass.
        
        Args:
            images: List of images to process
            
        Returns:
            descriptors: Array of shape (N, D) where N is number of images
        """
        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        images.to(device)
        
        with torch.no_grad():
            descriptors = self._model(images)
        
        # Need to unpack into a List of numpy arrays
        return [desc.detach().squeeze().cpu().numpy() for desc in descriptors]

    def describe(self, image: Image) -> np.ndarray:
        """
        Computes descriptor for a single image, applying its own transform.
        This is decoupled from the batch pipeline and is used by unit tests.
        """
        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        img_array = image.value_array.copy()
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        transform = self.get_preprocessing_transform()
        if transform is not None:
            img_tensor = transform(img_tensor)
        
        img_tensor = img_tensor.type(torch.float32) / 255.0
        img_tensor = img_tensor.to(device).unsqueeze(0)
        with torch.no_grad():
            descriptor = self._model(img_tensor)
            
        return descriptor.detach().squeeze().cpu().numpy()

    
