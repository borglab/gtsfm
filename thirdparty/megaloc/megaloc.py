"""Code for the MegaLoc model.
Much of the code in this file is from SALAD https://github.com/serizba/salad
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfm

# logger = logging.getLogger(__name__)
dinov2_path = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
if dinov2_path.exists() and str(dinov2_path) not in sys.path:
    sys.path.insert(0, str(dinov2_path))
    print(f"Added DINOv2 to sys.path (top level operation): {dinov2_path}")

class MegaLocModel(nn.Module):
    def __init__(
        self,
        feat_dim=8448,
        num_clusters=64,
        cluster_dim=256,
        token_dim=256,
        mlp_dim=512,
    ):
        super().__init__()
        self.backbone = DINOv2()
        self.salad_out_dim = num_clusters * cluster_dim + token_dim
        self.aggregator = Aggregator(
            feat_dim=feat_dim,
            agg_config={
                "num_channels": self.backbone.num_channels,
                "num_clusters": num_clusters,
                "cluster_dim": cluster_dim,
                "token_dim": token_dim,
                "mlp_dim": mlp_dim,
            },
            salad_out_dim=self.salad_out_dim,
        )
        self.feat_dim = feat_dim
        self.l2norm = L2Norm()

        # Load pretrained weights
        self.WEIGHT_URL = "https://github.com/gmberton/MegaLoc/releases/download/v1.0/megaloc.torch"
        self._load_pretrained_weights()

    def _load_pretrained_weights(self):

        """Load pretrained MegaLoc weights from GitHub release."""
        # Download and load pretrained weights
        #logger.info("Downloading MegaLoc weights from GitHub...")
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                self.WEIGHT_URL,
                map_location=torch.device("cpu"),
                progress=True  # Show download progress
            )
            self.load_state_dict(state_dict)
            #logger.info("✓ MegaLoc weights loaded successfully")
        except Exception as e:
            #logger.warning(f"Failed to download MegaLoc weights: {e}")
            #logger.warning("Model will use random initialization")
            print("Error {e}")

    def forward(self, images):
        b, c, h, w = images.shape
        if h % 14 != 0 or w % 14 != 0:
            # DINO needs height and width as multiple of 14, therefore resize them
            # to the nearest multiple of 14
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            images = tfm.functional.resize(images, [h, w], antialias=True)
        features = self.aggregator(self.backbone(images))
        features = self.l2norm(features)
        return features


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


class Aggregator(nn.Module):
    def __init__(self, feat_dim, agg_config, salad_out_dim):
        super().__init__()
        self.agg = SALAD(**agg_config)
        self.linear = nn.Linear(salad_out_dim, feat_dim)

    def forward(self, x):
        x = self.agg(x)
        return self.linear(x)


class DINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        
        dinov2_path = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
        if dinov2_path.exists() and str(dinov2_path) not in sys.path:
            sys.path.insert(0, str(dinov2_path))
            # logger.info(f"Added DINOv2 to sys.path: {dinov2_path}")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=False)
        self.num_channels = 768

    def forward(self, images):
        B, C, H, W = images.shape
        output = self.model.forward_features(images)
        cls_token = output["x_norm_clstoken"]
        features = output["x_norm_patchtokens"]
        features = features.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)
        return features, cls_token


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        mlp_dim=512,
        dropout=0.3,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.mlp_dim = mlp_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, self.mlp_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.mlp_dim, self.cluster_dim, 1),
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, self.mlp_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.mlp_dim, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        x (tuple): A tuple containing two elements, f and t.
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = x  # Extract features and token

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # Sinkhorn algorithm
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat(
            [
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )

        return nn.functional.normalize(f, p=2, dim=-1)