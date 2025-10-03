import itertools
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from custom_dataloader import DataParser as Parser
from custom_dataloader import Dataset
from gsplat import export_splats
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gtsfm.utils.splat import (
    get_viewmat,
    k_nearest_sklearn,
    num_sh_bases,
    random_quat_tensor,
    rescale_output_resolution,
    set_random_seed,
)


@dataclass
class Config:
    # --- Top-level settings ---
    ckpt: Optional[str] = None
    data_dir: str = "results/nerfstudio_input"
    images_dir: str = "tests/data/set1_lund_door/images/"
    result_dir: str = "results/custom_dataset"

    # --- Progressive Resolution Training ---
    num_downscales: int = 2
    resolution_schedule: int = 3000

    # --- Dataset and Global Settings ---
    test_every: int = 6
    normalize_world_space: bool = True
    batch_size: int = 1

    # --- Training Steps and Saving ---
    max_steps: int = 10000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = True
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # --- Gaussian Initialization ---
    init_type: Optional[str] = "random"
    init_num_pts: int = 100_000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1

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


def create_splats_with_optimizers(
    parser: Parser,
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
    ckpt: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:

    use_sfm_init = init_type == "sfm" and parser.points is not None and ckpt is None

    if use_sfm_init:
        print("Initializing from SfM points.")
        points = torch.from_numpy(parser.points).float()
        sh0_values = (torch.from_numpy(parser.point_colors).float() - 0.5) / 0.28209479177387814
        features_dc = torch.nn.Parameter(sh0_values)
    else:
        if init_type == "sfm":
            print("Warning: 'sfm' init chosen but no points found. Falling back to 'random'.")
        print(f"Initializing with {init_num_pts} random points.")
        random_scale = 10.0
        points = torch.nn.Parameter((torch.rand((init_num_pts, 3)) - 0.5) * random_scale)
        features_dc = torch.nn.Parameter(torch.rand(init_num_pts, 3))

    print("Calculating initial scales based on nearest neighbors...")
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

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

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


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config):

        set_random_seed(42)
        self.training = True

        self.cfg = cfg
        self.device = "cuda"

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        self.parser = Parser(
            data_dir=cfg.data_dir,
            images_dir=cfg.images_dir,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(self.parser, split="train")
        self.num_train_data = len(self.trainset)
        # print('Len of training data', self.num_train_data)
        self.valset = Dataset(self.parser, split="val")
        # print('Len of validation data', len(self.valset))

        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_opacity=cfg.init_opa,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            sh_degree=cfg.sh_degree,
            batch_size=cfg.batch_size,
            ckpt=cfg.ckpt,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        self.strategy = DefaultStrategy(
            prune_opa=cfg.cull_alpha_thresh,
            grow_grad2d=cfg.densify_grad_thresh,
            grow_scale3d=cfg.densify_size_thresh,
            grow_scale2d=cfg.split_screen_size,
            prune_scale3d=cfg.cull_scale_thresh,
            prune_scale2d=cfg.cull_screen_size,
            refine_scale2d_stop_iter=cfg.stop_screen_size_at,
            refine_start_iter=cfg.warmup_length,
            refine_stop_iter=cfg.stop_split_at,
            reset_every=cfg.reset_alpha_every * cfg.refine_every,
            refine_every=cfg.refine_every,
            pause_refine_after_reset=self.num_train_data + cfg.refine_every,
            absgrad=cfg.use_absgrad,
            revised_opacity=False,
            verbose=True,
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=False).to(self.device)

    def _get_downscale_factor(self, step: int) -> int:
        if not self.training:
            return 1
        num_doublings = max(0, step // self.cfg.resolution_schedule)
        factor = 2 ** max(self.cfg.num_downscales - num_doublings, 0)
        return int(factor)

    def _resize_image(self, image: torch.Tensor, d: int) -> torch.Tensor:
        if d == 1:
            return image
        image_permuted = image.permute(0, 3, 1, 2)
        in_channels = image_permuted.shape[1]
        weight = (1.0 / (d * d)) * torch.ones((in_channels, 1, d, d), dtype=torch.float32, device=image.device)
        downscaled = F.conv2d(image_permuted, weight, stride=d, groups=in_channels)
        return downscaled.permute(0, 2, 3, 1)

    def rasterize_splats(
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

    def train(self):
        self.training = True
        cfg = self.cfg
        device = self.device

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)),
        ]

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(itertools.cycle(trainloader))

        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)
            pixels_full_res = data["image"].to(device)
            Ks_full_res = data["K"].to(device)

            d = self._get_downscale_factor(step)
            if d > 1:
                height = math.floor(pixels_full_res.shape[1] * (1 / d))
                width = math.floor(pixels_full_res.shape[2] * (1 / d))
                pixels = self._resize_image(pixels_full_res, d)
                Ks = Ks_full_res.clone()
                Ks = rescale_output_resolution(Ks, 1 / d)
            else:
                height, width = pixels_full_res.shape[1:3]
                pixels = pixels_full_res
                Ks = Ks_full_res

            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # with autocast(enabled=True):
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
            )
            colors = renders[:, ..., :3]

            self.strategy.step_pre_backward(self.splats, self.optimizers, self.strategy_state, step, info)

            self.background_color = torch.tensor([0.1490, 0.1647, 0.2157])
            if cfg.random_bkgd and self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
            alphas = alphas[:, ...]
            # print('Colors after rasterizing in training', colors.shape)
            # print('Alphas after rasterizing in training', alphas.shape)
            colors = colors + background.unsqueeze(0).unsqueeze(0) * (1.0 - alphas)
            colors = torch.clamp(colors, 0.0, 1.0)

            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            loss.backward()
            pbar.set_description(f"loss={loss.item():.3f}| sh_deg={sh_degree_to_use}| res_factor={d}")

            # Unscale gradients and step optimizers
            for optimizer in self.optimizers.values():
                # scaler.step(optimizer)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()

            self.strategy.step_post_backward(
                self.splats, self.optimizers, self.strategy_state, step, info, packed=self.cfg.packed
            )

            if step in cfg.eval_steps or step == max_steps - 1:
                self.eval(step)
            if step in cfg.save_steps or step == max_steps - 1:
                data_to_save = {"step": step, "splats": self.splats.state_dict()}
                save_step = step
                if step == max_steps - 1:
                    save_step = step + 1
                torch.save(data_to_save, f"{self.ckpt_dir}/ckpt_{save_step}.pt")
            if step + 1 in cfg.ply_steps and cfg.save_ply:
                export_splats(
                    means=self.splats["means"],
                    scales=self.splats["scales"],
                    quats=self.splats["quats"],
                    opacities=self.splats["opacities"].squeeze(),
                    sh0=self.splats["sh0"],
                    shN=self.splats["shN"],
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step+1}.ply",
                )

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        if step == cfg.max_steps - 1:
            step = step + 1
        self.training = False

        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False)
        metrics = defaultdict(list)
        if len(valloader) == 0:
            return
        print(f"Running evaluation at step {step}...")

        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)

            pixels = data["image"].to(self.device)
            height, width = pixels.shape[1:3]

            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                render_mode="RGB+ED",
            )
            colors = colors[:, ..., :3]

            colors = torch.clamp(colors, 0.0, 1.0)
            os.makedirs(self.render_dir, exist_ok=True)

            img_to_save = (colors.squeeze().cpu().numpy() * 255).astype(np.uint8)

            output_filename = f"{self.render_dir}/{stage}_step{step}_{i:04d}.png"
            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_filename, img_to_save)

            gt_to_save = (pixels.squeeze().cpu().numpy() * 255).astype(np.uint8)

            gt_filename = f"{self.render_dir}/ground_truth_{i:04d}.png"
            gt_to_save = cv2.cvtColor(gt_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(gt_filename, gt_to_save)

            pixels_p = pixels.permute(0, 3, 1, 2)
            colors_p = colors.permute(0, 3, 1, 2)
            metrics["psnr"].append(self.psnr(colors_p, pixels_p))
            metrics["ssim"].append(self.ssim(colors_p, pixels_p))
            metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats["num_GS"] = len(self.splats["means"])
        print(f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}")

        with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        self.training = True


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    """Main training function."""
    if cfg.ckpt is not None:
        if not os.path.exists(cfg.ckpt):
            raise FileNotFoundError(f"Checkpoint file not found at {cfg.ckpt}")
        print(f"Loading checkpoint from {cfg.ckpt}")
        ckpt_data = torch.load(cfg.ckpt, map_location="cpu")

        num_points = ckpt_data["splats"]["means"].shape[0]
        print(f"Found {num_points} Gaussians in the checkpoint.")

        cfg.init_num_pts = num_points

    runner = Runner(cfg)

    if cfg.ckpt is not None:
        state_dict = {k: v.to(runner.device) for k, v in ckpt_data["splats"].items()}
        runner.splats.load_state_dict(state_dict)
        step = ckpt_data.get("step", 0)

        runner.eval(step=step + 1)
    else:
        runner.train()


if __name__ == "__main__":
    configs = {
        "default": (
            "Default configuration for training on a custom JSON-based dataset.",
            Config(),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)

    cli(main, cfg, verbose=True)
