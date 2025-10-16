""" Wrapper around the MegaLoc Global Descriptor
    Based on Gabrielle Berton's 
    "MegaLoc: One Retrieval to Place Them All"
    https://arxiv.org/pdf/2502.17237

    Authors: Kathir Gounder
"""
import numpy as np
import torch

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from thirdparty.megaloc.megaloc import MegaLocModel

class MegaLocGlobalDescriptor(GlobalDescriptorBase):
    def __init__(self) -> None:
        logger = logger_utils.get_logger()
        logger.info("⏳ Loading MegaLoc model weights...")
        self._model = MegaLocModel().eval()
        
    def describe(self, image: Image) -> np.ndarray:
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
        
        with torch.no_grad():
            descriptor = self._model(img_tensor)
            
        return descriptor.detach().squeeze().cpu().numpy()