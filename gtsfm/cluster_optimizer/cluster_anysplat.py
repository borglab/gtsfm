"""Feed forward Gaussian Splatting using the AnySplat model.
https://github.com/InternRobotics/AnySplat
Authors: Harneet Singh Khanuja
"""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Hashable, List

import cv2
import torch
import torchvision  # type: ignore
from dask import delayed  # type: ignore
from dask.delayed import Delayed

import gtsfm.frontend.anysplat as anysplat_utils
import gtsfm.utils.torch as torch_utils
from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterComputationGraph, ClusterContext, ClusterOptimizerBase
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.anysplat import AnySplatReconstructionResult
from gtsfm.ui.gtsfm_process import UiMetadata
from gtsfm.utils import logger as logger_utils

_SH0_NORMALIZATION_FACTOR = 0.28209479177387814

logger = logger_utils.get_logger()

# Module-level cache to reuse AnySplat weights per worker process.
_MODEL_CACHE: dict[Hashable, Any] = {}


class ClusterAnySplat(ClusterOptimizerBase):
    """Class for AnySplat (feed forward GS implementation)"""

    def __init__(
        self,
        model_loader: Callable[[], Any] | None = None,
        *,
        local_checkpoint: str | Path | None = None,
        model_cache_key: Hashable | None = None,
        max_num_points: int | None = None,
    ):
        """
        Initializes the ClusterAnySplat optimizer.
        Args:
            model_loader (Callable[[], Any] | None): Optional custom model loader function.
            local_checkpoint (str | Path | None): Local filesystem path to the model weights checkpoint.
            model_cache_key (Hashable | None): Key used to cache/reuse the loaded model across worker processes.
        """
        super().__init__()
        self._model = None
        self._device = torch_utils.default_device()
        self._max_gaussians = max_num_points
        if model_loader is not None:
            self._model_loader = model_loader
            self._model_cache_key = model_cache_key
        else:
            loader_kwargs: dict[str, Any] = {"device": self._device}
            if local_checkpoint is not None:
                checkpoint_path = local_checkpoint if isinstance(local_checkpoint, Path) else Path(local_checkpoint)
                loader_kwargs["checkpoint_path"] = checkpoint_path.expanduser()
            self._model_loader = partial(anysplat_utils.load_model, **loader_kwargs)
            if model_cache_key is not None:
                self._model_cache_key = model_cache_key
            else:
                checkpoint = loader_kwargs.get("checkpoint_path")
                self._model_cache_key = ("default_anysplat_loader", checkpoint)
        if not hasattr(self, "_model_cache_key"):
            self._model_cache_key = None

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""
        return UiMetadata(
            display_name="AnySplat",
            input_products=("Key Images",),
            output_products=("Gaussian Splats", "Interpolated Video"),
            parent_plate="Cluster Optimizer",
        )

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the AnySplat model to avoid unnecessary initialization."""
        if self._model is not None:
            return

        cache_key = getattr(self, "_model_cache_key", None)
        if cache_key is not None and cache_key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[cache_key]
            return

        logger.info("⏳ Loading AnySplat model weights...")
        model = self._model_loader()
        self._model = model
        if cache_key is not None:
            _MODEL_CACHE[cache_key] = model

    def __repr__(self) -> str:
        """Provide a readable summary of the optimizer configuration."""
        components = ["Model Name = AnySplat", "Input image preprocessed to (448,448)"]
        return "Feed Forward Gaussian Splatting(\n  " + ",\n  ".join(str(c) for c in components) + "\n)"

    def _aggregate_anysplat_metrics(self, result: AnySplatReconstructionResult) -> GtsfmMetricsGroup:
        """Capture simple runtime metrics for the front-end."""

        gaussian_count = result.gtsfm_data.number_images() * result.height * result.width
        voxel_count = result.splats.means.shape[1]
        return GtsfmMetricsGroup(
            "anysplat_runtime_metrics",
            [
                GtsfmMetric("total_pixel wise_gaussians", gaussian_count),
                GtsfmMetric("total_voxels", voxel_count),
                GtsfmMetric("Percent remaining after voxelize", voxel_count / gaussian_count),
            ],
        )

    def _generate_splats(self, images: dict[int, Image]) -> AnySplatReconstructionResult:
        """
        Apply AnySplat feed forward network to generate Gaussian splats.
        Args:
            images: Dictionary of images indexed by their global indices.

        Returns:
            an AnySplatReconstructionResult object
        """

        self._ensure_model_loaded()
        assert self._model is not None, "Model should be loaded by now"

        processed_images = self._preprocess_images(images, self._device)
        _, _, _, height, width = processed_images.shape
        logger.info("🔵 Running AnySplat on %d images.", len(images))
        splats, pred_context_pose = self._model.inference((processed_images + 1) * 0.5)

        # Move results to CPU to avoid Dask serialization errors.
        for attr in ["means", "scales", "rotations", "harmonics", "opacities", "covariances"]:
            setattr(splats, attr, getattr(splats, attr).cpu())
        pred_context_pose["extrinsic"] = pred_context_pose["extrinsic"].cpu()
        pred_context_pose["intrinsic"] = pred_context_pose["intrinsic"].cpu()
        decoder = self._model.decoder.cpu()

        gtsfm_data = GtsfmData(number_images=len(images))
        for local_idx, (global_idx, img) in enumerate(images.items()):
            intrinsic = pred_context_pose["intrinsic"][0][local_idx].numpy()
            extrinsic = pred_context_pose["extrinsic"][0][local_idx].numpy()
            camera = torch_utils.camera_from_matrices(extrinsic, intrinsic)
            gtsfm_data.add_camera(global_idx, camera)  # type: ignore
            gtsfm_data.set_image_info(
                global_idx,
                name=img.file_name,
                shape=(height, width),  # TODO(Frank): check if this is correct!
            )

        logger.info("Adding Gaussian means to GtsfmData as 3D tracks.")
        splats_means = splats.means[0].cpu().numpy()  # type: ignore
        dc_color = splats.harmonics[..., 0][0]  # type: ignore

        if self._max_gaussians is not None and splats.opacities.shape[1] > self._max_gaussians:
            logger.info(f"Filtering Gaussians to {self._max_gaussians}")

            op = splats.opacities[0]
            K = min(self._max_gaussians, op.numel())

            topk = torch.topk(op, k=K, largest=True, sorted=True)
            idx = topk.indices.to(splats.means.device).long()

            splats_means = splats.means[:, idx, :][0].cpu().numpy()

            dc_color = splats.harmonics[:, idx, :, :][..., 0][0]

        colors_tensor = (dc_color * _SH0_NORMALIZATION_FACTOR + 0.5).clamp(0.0, 1.0)
        colors_np = (colors_tensor * 255).cpu().numpy()

        if splats_means.size > 0:
            for j, xyz in enumerate(splats_means):
                color = colors_np[j]

                track = torch_utils.colored_track_from_point(xyz, color)

                gtsfm_data.add_track(track)

        logger.info(f"Added {len(splats_means)} tracks from Gaussian means.")

        gtsfm_data.set_gaussian_splats(splats)

        return AnySplatReconstructionResult(gtsfm_data, splats, pred_context_pose, height, width, decoder)

    def _preprocess_images(self, images: dict[int, Image], device):
        """
        Converts the data format from GtsfmData to AnySplat input
        """
        images_list = []
        for img in images.values():
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

    def _generate_interpolated_video(
        self,
        result: AnySplatReconstructionResult,
        save_gs_files_path: str,
    ) -> None:
        device = self._device
        if device == "cpu":
            logger.warning("CUDA not available, cannot generate interpolated video on CPU.")
            return

        # Move data to GPU
        decoder_model = result.decoder.to(device)
        extrinsics = result.pred_context_pose["extrinsic"].to(device)
        intrinsics = result.pred_context_pose["intrinsic"].to(device)
        for attr in ["means", "scales", "rotations", "harmonics", "opacities", "covariances"]:
            setattr(result.splats, attr, getattr(result.splats, attr).to(device))

        b = 1  # AnySplat convention
        anysplat_utils.save_interpolated_video(
            extrinsics,
            intrinsics,
            b,
            result.height,
            result.width,
            result.splats,
            save_gs_files_path,
            decoder_model,
        )

    def _save_splats(self, result: AnySplatReconstructionResult, save_gs_files_path: Path) -> None:
        splats = result.splats
        anysplat_utils.export_ply(
            splats.means[0],
            splats.scales[0],
            splats.rotations[0],
            splats.harmonics[0],
            splats.opacities[0],
            save_gs_files_path / "gaussian_splats.ply",
            save_sh_dc_only=True,  # Since current model use SH_degree = 4, which require large memory to store, we can
            # only save the DC band to save memory.
        )

    def create_computation_graph(
        self,
        context: ClusterContext,
    ) -> ClusterComputationGraph | None:
        """Create a Dask computation graph to process a cluster.

        Returns:
            a ClusterComputationGraph object
        """
        io_tasks: List[Delayed] = []
        metrics_tasks: List[Delayed] = []

        # Get images for all cluster indices as a delayed computation. within this cluster,
        d_cluster_images = context.get_delayed_image_map()
        if context.num_images == 0:
            raise ValueError("Cluster has no images to process.")

        result_graph = delayed(self._generate_splats)(d_cluster_images)

        metrics_tasks.append(delayed(self._aggregate_anysplat_metrics)(result_graph))
        with self._output_annotation():
            io_tasks.append(
                delayed(self._generate_interpolated_video)(
                    result_graph,
                    str(context.output_paths.results),
                )
            )
            io_tasks.append(delayed(self._save_splats)(result_graph, context.output_paths.results))

        sfm_result_graph = delayed(lambda res: res.gtsfm_data)(result_graph)

        return ClusterComputationGraph(
            io_tasks=tuple(io_tasks),
            metric_tasks=tuple(metrics_tasks),
            sfm_result=sfm_result_graph,
        )
