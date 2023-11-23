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

from gtsfm.common.image import Image
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from thirdparty.hloc.netvlad import NetVLAD


class NetVLADGlobalDescriptor(GlobalDescriptorBase):
    """NetVLAD global descriptor"""

    def __init__(self) -> None:
        """ """
        self._model = NetVLAD().eval()

    def describe(self, image: Image) -> np.ndarray:
        """Compute the NetVLAD global descriptor for a single image query.

        Args:
            image: Input image.

        Returns:
            img_desc: Array of shape (D,) representing global image descriptor.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        img_tensor = (
            torch.from_numpy(image.value_array).to(device).permute(2, 0, 1).unsqueeze(0).type(torch.float32) / 255
        )
        with torch.no_grad():
            img_desc = self._model({"image": img_tensor})

        return img_desc["global_descriptor"].detach().squeeze().cpu().numpy()
