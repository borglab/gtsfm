"""Training methods for Gaussian Splats

Authors: Harneet Singh Khanuja
"""

import itertools
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image

logger = logger_utils.get_logger()

_SH0_NORMALIZATION_FACTOR = 0.28209479177387814

# Size of the cube to initialize random gaussians within
# See https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py
RANDOM_SCALE = 10.0


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


if sys.platform == "darwin" or not torch.cuda.is_available():

    class GaussianSplatting:
        def __init__(self, *args, **kwargs):
            """Gaussian Splatting is not supported on Mac (darwin).
            Please run on Linux with CUDA and gsplat installed."""
            pass

        def rasterize_splats(self, *args, **kwargs):
            raise NotImplementedError("Gaussian Splatting is not supported on Mac (darwin).")

else:
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    from gtsfm.splat.gs_base import GSBase
    from gtsfm.splat.gs_data import GaussianSplattingData
    from gtsfm.utils import logger as logger_utils
    from gtsfm.utils.splat import (
        get_viewmat,
        k_nearest_sklearn,
        num_sh_bases,
        random_quat_tensor,
        rescale_output_resolution,
        set_random_seed,
    )

    class GaussianSplatting(GSBase):
        def __init__(self, cfg: Config, training: bool = True):

            set_random_seed(42)
            self.cfg = cfg
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.training = training
            self.strategy = DefaultStrategy(
                prune_opa=self.cfg.cull_alpha_thresh,
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
            full_dataset,
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
                full_dataset: Dataset class for Gaussian Splatting training data
                init_type: initialization from random points or sfm points
                init_num_pts: Number of randomly initialized gaussians
                init_opacity: Initial opacity value of the gaussians
                means_lr: learning rate for the means of the gaussians
                scales_lr: learning rate for the scales of the gaussians
                opacities_lr: learning rate for the opacities of the gaussians
                quats_lr: learning rate for the 3D rotation of the gaussians
                sh0_lr: learning rate for the 0th degree of the spherical harmonics of the gaussians
                shN_lr: learning rate for the all other degrees of the spherical harmonics of the gaussians
                sh_degree: degree for the spherical harmonics
                batch_size: batch size for the setup

            Returns:
                splats: Initialized 3D Gaussian Splats defining the entire scene
                optimizers: dictionary with optimizer parameters for different parameters of splats
            """

            use_sfm_init = init_type == "sfm" and full_dataset.points is not None

            if use_sfm_init:
                logger.info("Initializing from SfM points.")
                points = torch.from_numpy(full_dataset.points).float()
                sh0_values = (torch.from_numpy(full_dataset.point_colors).float() - 0.5) / _SH0_NORMALIZATION_FACTOR
                features_dc = torch.nn.Parameter(sh0_values)
            else:
                # See https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py
                if init_type == "sfm":
                    logger.info("Warning: 'sfm' init chosen but no points found. Falling back to 'random'.")
                logger.info("Initializing with %s random points.", init_num_pts)
                points = torch.nn.Parameter((torch.rand((init_num_pts, 3)) - 0.5) * RANDOM_SCALE)
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

            splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(self.device)

            optimizer_class = torch.optim.Adam

            optimizers = {
                name: optimizer_class(
                    [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
                    eps=1e-15 / math.sqrt(batch_size),
                    betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
                )
                for name, _, lr in params
            }

            return splats, optimizers

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

        def rasterize_splats(
            self,
            splats,
            wTi_tensor: Tensor,
            Ks: Tensor,
            width: int,
            height: int,
            render_mode: str,
            **kwargs,
        ) -> Tuple[Tensor, Tensor, Dict]:
            """
            Wrapper around the gsplat rasterization function

            Args:
                splats: 3D Gaussian splats
                wTi_tensor: camera-to-world matrices
                Ks: camera intrinsic matrices
                width: width of the rendered image
                height: height of the rendered image
                render_mode: rendering mode, 'RGB' or 'RGB+ED' for color and depth
                **kwargs: Additional arguments to pass to the rasterizer

            Returns:
                A tuple containing:
                    renders: rendered output tensor
                    alphas: rendered alpha channel
                    info: dictionary with internal data from the rasterizer
            """

            means = splats["means"]
            quats = splats["quats"]
            scales = torch.exp(splats["scales"])
            opacities = torch.sigmoid(splats["opacities"]).squeeze(-1)

            sh0, shN = splats["sh0"], splats["shN"]
            colors = torch.cat([sh0, shN], 1)

            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

            viewmats = get_viewmat(wTi_tensor)

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
                render_mode=render_mode,  # type: ignore
                **kwargs,
            )

        def _train(self, full_dataset, splats, optimizers):
            """
            The main training function which defines the dataloader
            and iterates max_steps times to refine the gaussian splats

            Args:
                full_dataset: Dataset class for Gaussian Splatting training data
                splats: Initialized 3D Gaussian Splats defining the entire scene
                optimizers: dictionary with optimizer parameters for different parameters of splats

            Returns:
                splats: finetuned 3D Gaussian Splats defining the entire scene
            """

            max_steps = self.cfg.max_steps
            init_step = 0

            schedulers = [
                torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / max_steps)),
            ]

            trainloader = torch.utils.data.DataLoader(
                full_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=0,
                persistent_workers=False,
                pin_memory=True,
            )
            trainloader_iter = iter(itertools.cycle(trainloader))

            for step in range(init_step, max_steps):
                data = next(trainloader_iter)

                wTi_tensor = data["wTc"].to(self.device)
                image_full_res = data["image"].to(self.device)
                Ks_full_res = data["K"].to(self.device)

                d = self._get_downscale_factor(step)
                if d > 1:
                    height = math.floor(image_full_res.shape[1] * (1 / d))
                    width = math.floor(image_full_res.shape[2] * (1 / d))
                    pixels = self._resize_image(image_full_res, d)
                    Ks = Ks_full_res.clone()
                    Ks = rescale_output_resolution(Ks, 1 / d)
                else:
                    height, width = image_full_res.shape[1:3]
                    pixels = image_full_res
                    Ks = Ks_full_res

                sh_degree_to_use = min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)

                renders, alphas, info = self.rasterize_splats(
                    splats=splats,
                    wTi_tensor=wTi_tensor,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=self.cfg.near_plane,
                    far_plane=self.cfg.far_plane,
                    render_mode="RGB",
                )
                colors = renders[:, ..., :3]

                self.strategy.step_pre_backward(splats, optimizers, self.strategy_state, step, info)

                background_color = torch.tensor([0.1490, 0.1647, 0.2157])
                if self.cfg.random_bkgd and self.training:
                    background = torch.rand(3, device=self.device)
                else:
                    background = background_color.to(self.device)
                alphas = alphas[:, ...]

                colors = colors + background.unsqueeze(0).unsqueeze(0) * (1.0 - alphas)
                colors = torch.clamp(colors, 0.0, 1.0)

                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - self.ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
                loss = l1loss * (1.0 - self.cfg.ssim_lambda) + ssimloss * self.cfg.ssim_lambda

                loss.backward()

                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                for scheduler in schedulers:
                    scheduler.step()

                self.strategy.step_post_backward(
                    splats, optimizers, self.strategy_state, step, info, packed=self.cfg.packed
                )

            return splats

        def splatify(self, images_graph: List[Image], sfm_result_graph: GtsfmData):
            """
            Main entry point to run Gaussian Splatting training and evaluation.

            Args:
                images_graph: List of images with Image object.
                sfm_result_graph: A GtsfmData object containing poses, points, etc.

            Returns:
                splats: finetuned 3D Gaussian Splats defining the entire scene
                cfg: Config file with training parameters
            """

            full_dataset = GaussianSplattingData(images_graph, sfm_result_graph)

            splats, optimizers = self._create_splats_with_optimizers(
                full_dataset=full_dataset,
                init_type=self.cfg.init_type,
                init_num_pts=self.cfg.init_num_pts,
                init_opacity=self.cfg.init_opacity,
                means_lr=self.cfg.means_lr,
                scales_lr=self.cfg.scales_lr,
                opacities_lr=self.cfg.opacities_lr,
                quats_lr=self.cfg.quats_lr,
                sh0_lr=self.cfg.sh0_lr,
                shN_lr=self.cfg.shN_lr,
                sh_degree=self.cfg.sh_degree,
                batch_size=self.cfg.batch_size,
            )

            splats = self._train(full_dataset, splats, optimizers)
            splats = {key: v.cpu() for key, v in splats.items()}

            return splats, self.cfg
