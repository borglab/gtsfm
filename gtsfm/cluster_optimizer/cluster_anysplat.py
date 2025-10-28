"""Feed forward Gaussians generator using the AnySplat model.
https://github.com/InternRobotics/AnySplat
Authors: Harneet Singh Khanuja
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import torch
import torchvision  # type: ignore
from dask import delayed
from dask.delayed import Delayed

from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterOptimizerBase
from gtsfm.common.image import Image
from gtsfm.products.visibility_graph import visibility_graph_keys
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


from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.model.ply_export import export_ply

logger = logger_utils.get_logger()


@dataclass
class FF_Config:
    """
    Parameters for Gaussian Splatting rendering
    """

    # --- Rendering ---
    save_video: bool = True
    save_ply_file: bool = True


class AnySplatGaussianSplatting(ClusterOptimizerBase):
    """Class for AnySplat (feed forward GS implementation)"""

    def __init__(self, cfg: FF_Config):
        """."""
        super().__init__()
        self.cfg = cfg
        self._model = None

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the AnySplat model to avoid unnecessary initialization."""
        if self._model is None:
            logger.info("⏳ Loading AnySplat model weights...")
            self._model = AnySplat.from_pretrained("lhjiang/anysplat")

    def generate_splats(
        self,
        images: List[Image],
        save_gs_files_path: Path,
    ) -> None:
        """
        Apply AnySplat feed forward network to generate Gaussian splats.
        Args:
            images: List of all images.
            save_gs_files_path: Path to save ply file with all Gaussians
        Returns:
        """

        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        self._apply_anysplat(self._model, images, self.cfg, save_gs_files_path)

        return

    def _preprocess_images(self, images: List[Image], device):
        """
        Converts the data format from GtsfmData to AnySplat input
        """

        images_list = []
        for i, img in enumerate(images):
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

    def _apply_anysplat(self, model, images: List[Image], cfg, save_gs_files_path):
        """
        Function that runs the inference on feed forward GS models and writes results to disk
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        processed_images = self._preprocess_images(images, device)
        splats, pred_context_pose = model.inference((processed_images + 1) * 0.5)

        if cfg.save_video:
            b, v, _, height, width = processed_images.shape
            self._generate_interpolated_video(
                pred_context_pose["extrinsic"],
                pred_context_pose["intrinsic"],
                b,
                height,
                width,
                splats,
                str(save_gs_files_path),
                model.decoder,
            )

        if cfg.save_ply_file:
            self._save_splats(splats, save_gs_files_path)

        return

    def _generate_interpolated_video(
        self, extrinsics, intrinsics, b, height, width, splats, save_gs_files_path, decoder_model
    ):
        save_interpolated_video(
            extrinsics,
            intrinsics,
            b,
            height,
            width,
            splats,
            save_gs_files_path,
            decoder_model,
        )

    def _save_splats(self, splats, save_gs_files_path):
        export_ply(
            splats.means[0],
            splats.scales[0],
            splats.rotations[0],
            splats.harmonics[0],
            splats.opacities[0],
            save_gs_files_path / "gaussian_splats.ply",
        )

    def create_computation_graph(
        self,
        num_images: int,
        one_view_data_dict,
        output_paths,
        loader,
        output_root: Path,
        visibility_graph,
        image_futures,
    ) -> tuple[list[Delayed], list[Delayed]]:
        """Create a Dask computation graph to process a cluster.

        Returns:
            - List of Delayed I/O tasks to be computed
            - List of Delayed metrics to be computed
        """
        io_tasks: List[Delayed] = []
        metrics_tasks: List[Delayed] = []

        keys = sorted(visibility_graph_keys(visibility_graph))

        def _pack_images_for_anysplat(*images: Image) -> list[Image]:
            """Collect variadic image inputs into an ordered list."""

            return list(images)

        # TODO(Frank): see if you could ask the loader for a batch (in a delayed way)
        # Something like: loader.get_images_as_delayed_batch(keys)
        delayed_images_map = loader.get_images_as_delayed_map()
        selected_images = [delayed_images_map[key] for key in keys if key in delayed_images_map]

        images_graph = delayed(_pack_images_for_anysplat)(*selected_images)

        # TODO(Harneet): separate compute and writing
        with self._output_annotation():
            io_tasks.append(delayed(self.generate_splats)(images_graph, output_paths.gs_path))

        return io_tasks, metrics_tasks
