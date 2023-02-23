"""Source: Adapted from "Hierarchical-Localization" toolbox.
https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/extractors/netvlad.py

Author: Paul-Edouard Sarlin
"""

import subprocess
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

logger = logging.getLogger(__name__)

# path to /thirdparty/hloc/weights/{CHECKPOINT}.mat
netvlad_path = Path(__file__).resolve().parent / "weights"

EPS = 1e-6


class NetVLADLayer(nn.Module):
    """Computes the 'Vector of Locally Aggregated Descriptors' in a differentiable fashion.

    Whereas bag-of-visual-words aggregation keeps counts of visual words, VLAD stores the sum of residuals
    (difference vector between the descriptor and its corresponding cluster centre) for each visual word.
    """
    def __init__(self, input_dim: int = 512, K: int = 64, score_bias: bool = False, intranorm: bool = True) -> None:
        """
        Args:
            input_dim: output feature map from fully-convolutional backbone has shape (input_dim,H2,W2) 
            K: number of cluster centers.
            score_bias: whether to use bias term in 1x1 conv (projection operation).
            intranorm: whether to normalize descriptors immediately after computing sum of residuals.
        """
        super().__init__()
        self.score_proj = nn.Conv1d(input_dim, K, kernel_size=1, bias=score_bias)
        centers = nn.parameter.Parameter(torch.empty([input_dim, K]))
        nn.init.xavier_uniform_(centers)
        self.register_parameter("centers", centers)
        self.intranorm = intranorm
        self.output_dim = input_dim * K

    def forward(self, x: Tensor) -> Tensor:
        """Given output of fully-convolutional backbone, compute sum of residuals against cluster centers.
        Cluster centers represent visual words.

        Args:
            x: tensor of shape (1, 512, H2 * W2]) where (C2,H2,W2) is the size of the output feature map.

        Returns:
            desc: tensor of shape (1,32768), as (512 * 64) is reshaped to (1,32768).
        """
        b = x.size(0)
        # (1,512,2961) -> (1,64,2961)
        scores = self.score_proj(x)
        scores = F.softmax(scores, dim=1)
        diff = x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1)
        desc = (scores.unsqueeze(1) * diff).sum(dim=-1)
        if self.intranorm:
            # From the official MATLAB implementation.
            desc = F.normalize(desc, dim=1)
        desc = desc.view(b, -1)
        desc = F.normalize(desc, dim=1)
        return desc


class NetVLAD(nn.Module):
    default_conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "checkpoint_dir": netvlad_path, "whiten": True}
    required_inputs = ["image"]

    # Models exported using
    # https://github.com/uzh-rpg/netvlad_tf_open/blob/master/matlab/net_class2struct.m.
    dir_models = {
        "VGG16-NetVLAD-Pitts30K": "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat",
        "VGG16-NetVLAD-TokyoTM": "https://cvg-data.inf.ethz.ch/hloc/netvlad/TokyoTM_struct.mat",
    }

    def __init__(self, conf: Dict[str, Any] = default_conf):
        """
        Args:
            config: model config w/ inference settings.
        """
        super().__init__()
        assert conf["model_name"] in self.dir_models.keys()

        # Download the checkpoint.
        checkpoint = conf["checkpoint_dir"] / str(conf["model_name"] + ".mat")
        if not checkpoint.exists():
            checkpoint.parent.mkdir(exist_ok=True)
            link = self.dir_models[conf["model_name"]]
            cmd = ["wget", link, "-O", str(checkpoint)]
            logger.info(f"Downloading the NetVLAD model with `{cmd}`.")
            subprocess.run(cmd, check=True)

        # Create the network.
        # Remove classification head.
        backbone = list(models.vgg16().children())[0]
        # Remove last ReLU + MaxPool2d.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.netvlad = NetVLADLayer()

        if conf["whiten"]:
            self.whiten = nn.Linear(self.netvlad.output_dim, 4096)

        # Parse MATLAB weights using https://github.com/uzh-rpg/netvlad_tf_open
        print("loading ", checkpoint)
        mat = scipy.io.loadmat(checkpoint, struct_as_record=False, squeeze_me=True)

        # CNN weights.
        for layer, mat_layer in zip(self.backbone.children(), mat["net"].layers):
            if isinstance(layer, nn.Conv2d):
                w = mat_layer.weights[0]  # Shape: S x S x IN x OUT
                b = mat_layer.weights[1]  # Shape: OUT
                # Prepare for PyTorch - enforce float32 and right shape.
                # w should have shape: OUT x IN x S x S
                # b should have shape: OUT
                w = torch.tensor(w).float().permute([3, 2, 0, 1])
                b = torch.tensor(b).float()
                # Update layer weights.
                layer.weight = nn.Parameter(w)
                layer.bias = nn.Parameter(b)

        # NetVLAD weights.
        score_w = mat["net"].layers[30].weights[0]  # D x K
        # centers are stored as opposite in official MATLAB code
        center_w = -mat["net"].layers[30].weights[1]  # D x K
        # Prepare for PyTorch - make sure it is float32 and has right shape.
        # score_w should have shape K x D x 1
        # center_w should have shape D x K
        score_w = torch.tensor(score_w).float().permute([1, 0]).unsqueeze(-1)
        center_w = torch.tensor(center_w).float()
        # Update layer weights.
        self.netvlad.score_proj.weight = nn.Parameter(score_w)
        self.netvlad.centers = nn.Parameter(center_w)

        # Whitening weights.
        if conf["whiten"]:
            w = mat["net"].layers[33].weights[0]  # Shape: 1 x 1 x IN x OUT
            b = mat["net"].layers[33].weights[1]  # Shape: OUT
            # Prepare for PyTorch - make sure it is float32 and has right shape
            w = torch.tensor(w).float().squeeze().permute([1, 0])  # OUT x IN
            b = torch.tensor(b.squeeze()).float()  # Shape: OUT
            # Update layer weights.
            self.whiten.weight = nn.Parameter(w)
            self.whiten.bias = nn.Parameter(b)

        # Preprocessing parameters.
        self.preprocess = {
            "mean": mat["net"].meta.normalization.averageImage[0, 0],
            "std": np.array([1, 1, 1], dtype=np.float32),
        }

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            data: dictionary with key "image" and value (1,3,H,W) tensor.

        Returns:
            dictionary containing key "global_descriptor", value (1,4096)
        """
        image = data["image"]
        assert image.shape[1] == 3
        assert image.min() >= -EPS and image.max() <= 1 + EPS
        image = torch.clamp(image * 255, 0.0, 255.0)  # Input should be 0-255.
        mean = self.preprocess["mean"]
        std = self.preprocess["std"]
        image = image - image.new_tensor(mean).view(1, -1, 1, 1)
        image = image / image.new_tensor(std).view(1, -1, 1, 1)

        # Feature extraction.
        descriptors = self.backbone(image)
        b, c, _, _ = descriptors.size()
        descriptors = descriptors.view(b, c, -1)

        # NetVLAD layer.
        descriptors = F.normalize(descriptors, dim=1)  # Pre-normalization to unit-length.
        desc = self.netvlad(descriptors)

        # Whiten if needed.
        if hasattr(self, "whiten"):
            desc = self.whiten(desc)
            desc = F.normalize(desc, dim=1)  # Final L2 normalization.

        return {"global_descriptor": desc}
