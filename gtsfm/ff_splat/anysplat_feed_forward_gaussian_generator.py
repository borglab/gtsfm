"""Feed forward Gaussians generator using the AnySplat model.

https://github.com/InternRobotics/AnySplat

Authors: Harneet Singh Khanuja
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torchvision

from gtsfm.ff_splat.gaussian_generator_base import FeedForwardGaussianGeneratorBase
from gtsfm.splat import rendering
from gtsfm.utils import logger as logger_utils

HERE_PATH = Path(__file__).parent
ANYSPLAT_REPO_PATH = HERE_PATH.parent.parent / "thirdparty" / "AnySplat"

if ANYSPLAT_REPO_PATH.exists():
    # workaround for sibling import
    sys.path.insert(0, str(ANYSPLAT_REPO_PATH))
else:
    raise ImportError(
        f"mast3r is not initialized, could not find: {ANYSPLAT_REPO_PATH}.\n "
        "Did you forget to run 'git submodule update --init --recursive' ?"
    )


from dask.distributed import Future, worker_client
from gsplat import export_splats
from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.model.ply_export import export_ply

from gtsfm.common.image import Image

logger = logger_utils.get_logger()


@dataclass
class FF_Config:
    """
    Parameters for Gaussian Splatting rendering
    """

    # --- Rendering ---
    sh_degree: int = 4
    save_video: bool = True
    fps: int = 30
    num_frames: int = 20
    save_ply_file: bool = True


class AnySplatGaussianGenerator(FeedForwardGaussianGeneratorBase):

    def __init__(self, cfg: FF_Config):
        """."""
        super().__init__()
        self.cfg = cfg

    def generate_splats(
        self,
        images: List[Future],
        save_gs_files_path: Path,
    ) -> None:
        """
        Apply AnySplat feed forward network to generate Gaussian splats.

        Args:
            images: List of all images, as futures.
            save_gs_files_path: Path to save ply file with all Gaussians

        Returns:
        """
        with worker_client() as gsclient:
            logger.info("⏳ Loading AnySplat model weights...")
            model = AnySplat.from_pretrained("lhjiang/anysplat")

            m = gsclient.scatter(model, broadcast=False)
            cfg = self.cfg

            splat_futures = gsclient.submit(
                AnySplatGaussianGenerator.apply_anysplat, m, images, cfg, save_gs_files_path
            )
            splat_futures.result()

            return

    @staticmethod
    def preprocess_images(images: List[Future], device):
        """
        Converts the data format from GtsfmData to AnySplat input
        """

        images_list = []
        for i, img in enumerate(images):
            if i % 4 != 0:
                continue
            height, width = img.shape[:2]
            img_array = img.value_array
            if width > height:
                new_height = 448
                new_width = int(width * (new_height / height))
            else:
                new_width = 448
                new_height = int(height * (new_width / width))
            img_array = cv2.resize(img_array, (new_width, new_height))

            # Center crop
            left = (new_width - 448) // 2
            top = (new_height - 448) // 2
            right = left + 448
            bottom = top + 448
            img_array = img_array[top:bottom, left:right]  # cropping
            img_tensor = torchvision.transforms.ToTensor()(img_array) * 2.0 - 1.0  # [-1, 1]
            images_list.append(img_tensor.to(device))
        images_tensor = torch.stack(images_list, dim=0).unsqueeze(0)
        return images_tensor.to(device)

    @staticmethod
    def apply_anysplat(model, images: List[Future], cfg, save_gs_files_path):
        """
        Function that runs the inference on feed forward GS models and writes results to disk
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        processed_images = AnySplatGaussianGenerator.preprocess_images(images, device)
        splats, pred_context_pose = model.inference((processed_images + 1) * 0.5)

        if cfg.save_video:
            b, v, _, height, width = processed_images.shape
            # save_interpolated_video(
            #     pred_context_pose["extrinsic"],
            #     pred_context_pose["intrinsic"],
            #     b,
            #     height,
            #     width,
            #     splats,
            #     str(save_gs_files_path),
            #     model.decoder,
            # )
            AnySplatGaussianGenerator.generate_interpolated_video(
                model.decoder,
                cfg,
                splats,
                pred_context_pose,
                height,
                width,
                save_gs_files_path / "interpolated_video.mp4",
                device,
            )

        if cfg.save_ply_file:
            export_ply(
                splats.means[0],
                splats.scales[0],
                splats.rotations[0],
                splats.harmonics[0],
                splats.opacities[0],
                save_gs_files_path / "gaussian_splats.ply",
            )

        return

    @staticmethod
    def generate_interpolated_video(model_decoder, cfg, splats, context_poses, height, width, video_fpath, device):
        wTi_np = rendering.generate_interpolated_path(
            context_poses["extrinsic"][0, :, :3, :].cpu().numpy(), cfg.num_frames, spline_degree=2
        )

        wTi_np = np.concatenate(
            [
                wTi_np,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(wTi_np), axis=0),
            ],
            axis=1,
        )
        wTi_tensor = torch.from_numpy(wTi_np).float().unsqueeze(0).to(device)

        # Interpolate intrinsics to match the number of interpolated extrinsics
        num_original_frames = context_poses["intrinsic"].shape[1]
        num_interpolated_frames = wTi_tensor.shape[1]

        if num_original_frames != num_interpolated_frames:
            original_intrinsics = context_poses["intrinsic"].cpu().numpy()
            interpolated_intrinsics = np.zeros((num_interpolated_frames, 3, 3), dtype=np.float32)

            for i in range(num_interpolated_frames):
                # Calculate the corresponding index in the original frames
                # This is a simple linear mapping
                original_idx_float = (i / (num_interpolated_frames - 1)) * (num_original_frames - 1)
                original_idx_floor = int(np.floor(original_idx_float))
                original_idx_ceil = int(np.ceil(original_idx_float))

                if original_idx_floor == original_idx_ceil:
                    interpolated_intrinsics[i] = original_intrinsics[0, original_idx_floor]
                else:
                    alpha = original_idx_float - original_idx_floor
                    interp_k = (1 - alpha) * original_intrinsics[0, original_idx_floor] + alpha * original_intrinsics[
                        0, original_idx_ceil
                    ]
                    interpolated_intrinsics[i] = interp_k

            K = torch.from_numpy(interpolated_intrinsics).float().unsqueeze(0).to(device)
        else:
            K = context_poses["intrinsic"].to(device)

        logger.info("%s", wTi_tensor.shape)
        logger.info("%s", K.shape)
        interpolated_output = model_decoder.forward(
            splats,
            wTi_tensor,
            K,
            torch.ones(1, K.shape[1], device=wTi_tensor.device) * 0.1,
            torch.ones(1, K.shape[1], device=wTi_tensor.device) * 100,
            (height, width),
        )

        color_output = interpolated_output.color[0].clip(min=0, max=1)
        color_output = color_output.cpu().detach().numpy()
        color_output = np.transpose(color_output, (0, 2, 3, 1))
        color_output = (color_output * 255).astype(np.uint8)

        if width <= 0 or height <= 0:
            logger.error("Invalid frame size: width=%s, height=%s. Cannot create video writer.", width, height)
            return

        frame_size = (width, height)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        try:
            writer = cv2.VideoWriter(str(video_fpath), fourcc, cfg.fps, frame_size)
        except cv2.error as e:
            logger.error("Error initializing VideoWriter: %s", e)
            return

        for canvas in color_output:
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            writer.write(canvas_bgr)
        writer.release()
        logger.info("Interpolated video saved to %s", video_fpath)
