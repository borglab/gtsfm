"""Training methods for Gaussian Splats

Authors: Harneet Singh Khanuja
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from gtsfm.splat.gs_data import GaussianSplattingData

from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.splat.gs_base import GSBase

from gtsfm.utils.splat import (
    rescale_output_resolution, 
    k_nearest_sklearn, 
    random_quat_tensor, 
    num_sh_bases, 
    get_viewmat, 
    set_random_seed
)
from gtsfm.utils import logger as logger_utils


logger = logger_utils.get_logger()

@dataclass
class Config:
    """
    Parameters for Gaussian Splatting training
    """
    # --- Progressive Resolution Training ---
    num_downscales: int = 2
    resolution_schedule: int = 3000
    
    # --- Dataset Settings ---
    batch_size: int = 1

    # --- Training Steps  ---
    max_steps: int = 7000

    # --- Gaussian Initialization ---
    init_type: str = "sfm"
    init_num_pts: int = 100_000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opacity: float = 0.1

    # --- Loss and Regularization ---
    ssim_lambda: float = 0.2
    random_bkgd: bool = True 

    # --- Rendering Parameters ---
    near_plane: float = 0.01
    far_plane: float = 1e10
    antialiased: bool = False
    packed: bool = False

    # --- Advanced Densification and Pruning Strategy Controls (from Splatfacto) ---
    warmup_length: int = 500
    refine_every: int = 100
    cull_alpha_thresh: float = 0.005
    cull_scale_thresh: float = 0.2
    reset_alpha_every: int = 30
    densify_grad_thresh: float = 0.0002
    densify_size_thresh: float = 0.01
    cull_screen_size: float = 0.15
    split_screen_size: float = 0.05
    stop_screen_size_at: int = 4000
    stop_split_at: int = 3000
    n_split_samples: int = 2
    use_absgrad: bool = True

    # --- Learning Rates (aligned with splatfacto) ---
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    # --- Rendering ---
    fps: int = 30
    num_frames: int = 10

class GaussianSplatting(GSBase):
    def __init__(self, cfg: Config):
        set_random_seed(42)
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.training = True
        self.strategy = DefaultStrategy(
            prune_opa= self.cfg.cull_alpha_thresh,
            grow_grad2d=self.cfg.densify_grad_thresh,
            grow_scale3d=self.cfg.densify_size_thresh,
            grow_scale2d=self.cfg.split_screen_size,
            prune_scale3d=self.cfg.cull_scale_thresh,
            prune_scale2d=self.cfg.cull_screen_size,
            refine_scale2d_stop_iter=self.cfg.stop_screen_size_at,
            refine_start_iter=self.cfg.warmup_length,
            refine_stop_iter=self.cfg.stop_split_at,
            reset_every=self.cfg.reset_alpha_every * self.cfg.refine_every,
            absgrad=self.cfg.use_absgrad,
            revised_opacity=False,
            verbose=True,
            )
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        
    def _create_splats_with_optimizers(
        self,
        init_type: str = "random",
        init_num_pts: int = 100_000,
        init_opacity: float = 0.1,
        means_lr: float = 1.6e-4,
        scales_lr: float = 5e-3,
        opacities_lr: float = 5e-2,
        quats_lr: float = 1e-3,
        sh0_lr: float = 2.5e-3,
        shN_lr: float = 2.5e-3 / 20,
        sh_degree: int = 3,
        batch_size: int = 1,
    ):
        
        """
        Create the initial Gaussian Splats with parameters defined in the Config class
        Args:
            init_type: intialization from random points or sfm points
            init_num_pts: Number of randomly initialized gaussians
            init_opacity: Initial opacity value of the gaussians
            means_lr: learning rate for the means of the gaussians
            scales_lr: learning rate for the means of the gaussians
            opacities_lr: learning rate for the opacities of the gaussians
            quats_lr: learning rate for the 3D rotation (covariance) of the gaussians
            sh0_lr: learning rate for the 0th degree of the spherical harmonics of the gaussians
            shN_lr: learning rate for the all other degrees of the spherical harmonics of the gaussians
            sh_degree: degree for the spherical harmonics
            batch_size: batch size for the setup
        """
        
        use_sfm_init = (init_type == "sfm" and self.full_dataset.points is not None)
        
        if use_sfm_init:
            logger.info("Initializing from SfM points.")
            points = torch.from_numpy(self.full_dataset.points).float()
            sh0_values = (torch.from_numpy(self.full_dataset.point_colors).float() - 0.5) / 0.28209479177387814
            features_dc = torch.nn.Parameter(sh0_values)
        else:
            if init_type == "sfm":
                logger.info("Warning: 'sfm' init chosen but no points found. Falling back to 'random'.")
            logger.info(f"Initializing with {init_num_pts} random points.")
            random_scale = 10.0
            points = torch.nn.Parameter((torch.rand((init_num_pts, 3)) - 0.5) * random_scale)
            features_dc = torch.nn.Parameter(torch.rand(init_num_pts, 3))

        logger.info("Calculating initial scales based on nearest neighbors...")
        distances, _ = k_nearest_sklearn(points.data, 3)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))

        num_points = points.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(sh_degree)

        
        features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(init_opacity * torch.ones(num_points, 1)))

        params = [
            ("means", points, means_lr),
            ("scales", scales, scales_lr),
            ("quats", quats, quats_lr),
            ("opacities", opacities, opacities_lr),
        ]

        initial_colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
        params.append(("sh0", initial_colors[:, :1, :], sh0_lr))
        params.append(("shN", initial_colors[:, 1:, :], shN_lr))

        self.splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(self.device)
        
        self.optimizer_class = torch.optim.Adam
        
        self.optimizers = {
            name: self.optimizer_class(
                [{"params": self.splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
            )
            for name, _, lr in params
        }

    def _get_downscale_factor(self, step: int) -> int:
        """
        Get the factor by how much the image and intrinsics should be rescaled for progressive resolution training
        Args:
            step: the current step iteration of the training
        
        Returns:
            factor: the factor for rescaling image and intrinsics
        """
        if not self.training:
            return 1
        num_doublings = max(0, step // self.cfg.resolution_schedule)
        factor = 2 ** max(self.cfg.num_downscales - num_doublings, 0)
        return int(factor)

    def _resize_image(self, image: torch.Tensor, d: int) -> torch.Tensor:
        """
        Rescaling the image for progressive resolution training
        Args:
            image: the image to be rescaled
            d: the rescale factor
        
        Returns:
            downscaled_image: the resized image
        """
        if d == 1:
            return image
        image_permuted = image.permute(0, 3, 1, 2)
        in_channels = image_permuted.shape[1]
        weight = (1.0 / (d * d)) * torch.ones((in_channels, 1, d, d), dtype=torch.float32, device=image.device)
        downscaled = F.conv2d(image_permuted, weight, stride=d, groups=in_channels)
        return downscaled.permute(0, 2, 3, 1)

    def _rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        render_mode: str,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)
        
        sh0, shN = self.splats["sh0"], self.splats["shN"]
        colors = torch.cat([sh0, shN], 1)

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        
        viewmats = get_viewmat(camtoworlds)

        return rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.strategy.absgrad,
            rasterize_mode=rasterize_mode,
            render_mode=render_mode,
            **kwargs,
        )
    
    def _train(self):
        """
        The main training function which defines the dataloader 
        and iterates max_steps times to refine the gaussian splats
        """
        
        max_steps = self.cfg.max_steps
        init_step = 0

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        trainloader = torch.utils.data.DataLoader(
            self.full_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
            pin_memory=True,
        )
        trainloader_iter = iter(itertools.cycle(trainloader)) 

        for step in range(init_step, max_steps):
            data = next(trainloader_iter)
            
            camtoworlds = data["camtoworld"].to(self.device)
            image_full_res = data["image"].to(self.device)
            Ks_full_res = data["K"].to(self.device)

            d = self._get_downscale_factor(step)
            if d > 1:
                height = math.floor(image_full_res.shape[1] * (1/d))
                width = math.floor(image_full_res.shape[2] * (1/d))
                pixels = self._resize_image(image_full_res, d)
                Ks = Ks_full_res.clone()
                Ks = rescale_output_resolution(Ks, 1/d)
            else:
                height, width = image_full_res.shape[1:3]
                pixels = image_full_res
                Ks = Ks_full_res
            
            sh_degree_to_use = min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)

            renders, alphas, info = self._rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                render_mode="RGB"
            )
            colors = renders[:,..., :3]

            self.strategy.step_pre_backward(
                self.splats, self.optimizers, self.strategy_state, step, info
            )

            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )
            if self.cfg.random_bkgd and self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
            alphas = alphas[:, ...]

            colors = colors + background.unsqueeze(0).unsqueeze(0) * (1.0 - alphas)
            colors = torch.clamp(colors, 0.0, 1.0)

            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 -self. cfg.ssim_lambda) + ssimloss * self.cfg.ssim_lambda

            loss.backward()

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()

            self.strategy.step_post_backward(
                self.splats, self.optimizers, self.strategy_state, step, info,
                packed=self.cfg.packed
            )

    def splatify(self, images_graph: Dict[int, Image], sfm_result_graph: GtsfmData):
        """
        Main entry point to run Gaussian Splatting training and evaluation.

        Args:
            images_graph: A dictionary mapping indices to Image objects.
            sfm_result_graph: A GtsfmData object containing poses, points, etc.

        Returns:
            splats: 3D Gaussian Splats defining the entire scene
            cfg: Config file with training parameters
        """

        self.full_dataset = GaussianSplattingData(images_graph, sfm_result_graph)
        
        
        self._create_splats_with_optimizers(
            init_type = self.cfg.init_type,
            init_num_pts = self.cfg.init_num_pts,
            init_opacity = self.cfg.init_opacity,
            means_lr = self.cfg.means_lr,
            scales_lr = self.cfg.scales_lr,
            opacities_lr = self.cfg.opacities_lr,
            quats_lr = self.cfg.quats_lr,
            sh0_lr = self.cfg.sh0_lr,
            shN_lr = self.cfg.shN_lr,
            sh_degree = self.cfg.sh_degree,
            batch_size = self.cfg.batch_size,
        )

        self._train()

        return self.splats, self.cfg