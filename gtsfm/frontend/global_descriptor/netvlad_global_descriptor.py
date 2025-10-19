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

from typing import List, Optional

import numpy as np
import torch

import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from thirdparty.hloc.netvlad import NetVLAD


class NetVLADGlobalDescriptor(GlobalDescriptorBase):
    """NetVLAD global descriptor"""

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[torch.nn.Module] = None  # Lazy loading - only load when describe() is called

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the NetVLAD model to avoid unnecessary initialization."""
        if self._model is None:
            logger = logger_utils.get_logger()
            logger.info("⏳ Loading NetVLAD model weights...")
            self._model = NetVLAD().eval()

    def describe_batch(self, images: List[Image]) -> List[np.ndarray]:
        """Compute descriptors for a batch of images efficiently."""
        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        # 1. Convert all images in the list to a single batch tensor.
        tensors = [
            torch.from_numpy(img.value_array.copy()).permute(2, 0, 1).to(device)
            for img in images
        ]
        batch_tensor = torch.stack(tensors).type(torch.float32) / 255.0
        
        with torch.no_grad():
            # 2. Get all descriptors from the model in a single forward pass.
            batch_descriptors = self._model({"image": batch_tensor})

        # 3. Convert the output tensor back to a list of numpy arrays.
        descs_np = batch_descriptors["global_descriptor"].detach().cpu().numpy()
        return [desc for desc in descs_np]
    
    def describe(self, image: Image) -> np.ndarray:
        """Compute descriptor for a single image (delegates to batch method)."""
        return self.describe_batch([image])[0]
