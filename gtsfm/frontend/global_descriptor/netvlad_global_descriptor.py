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

from typing import List, Optional, Callable

import numpy as np
import torch
from torchvision import transforms

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

    def get_preprocessing_transform(self) -> Optional[Callable]:
        """"Return transform to resize images to 480x640 (height x width).
    
            NetVLAD operates on convolutional feature maps and doesn't require 
            the original VGG16 input size (224x224). Research implementations 
            commonly use ~480x640 for a good balance of descriptor quality and 
            memory efficiency during batching.
        """
        return transforms.Resize(size=(480, 640), antialias=True)

    def describe_batch(self, images: torch.Tensor) -> List[np.ndarray]:
        """Compute descriptors for a batch of images efficiently."""
        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        
        with torch.no_grad():
            # 2. Get all descriptors from the model in a single forward pass.
            batch_descriptors = self._model({"image": images})

        # 3. Convert the output tensor back to a list of numpy arrays.
        descs_np = batch_descriptors["global_descriptor"].detach().cpu().numpy()
        return [desc for desc in descs_np]
    
    def describe(self, image: Image) -> np.ndarray:
        """"""
        # Convert Image to tensor
        image_array = image.value_array
        if isinstance(image_array, np.ndarray):
            image_tensor = torch.from_numpy(image_array).float()
        else:
            image_tensor = image_array
        
        # Apply preprocessing transform (resize to 480x640)
        transform = self.get_preprocessing_transform()
        if transform is not None:
            image_tensor = transform(image_tensor)
        
        # Add batch dimension
        batch_tensor = image_tensor.unsqueeze(0)
        return self.describe_batch(batch_tensor)[0]
