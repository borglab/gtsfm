"""VGGT-based cluster optimizer leveraging the demo VGGT pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from dask.delayed import Delayed, delayed

import gtsfm.utils.vggt as vggt
from gtsfm.cluster_optimizer.cluster_optimizer_base import (
    REACT_RESULTS_PATH,
    ClusterComputationGraph,
    ClusterContext,
    ClusterOptimizerBase,
)
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.products.visibility_graph import visibility_graph_keys
from gtsfm.ui.gtsfm_process import UiMetadata
from gtsfm.utils.logger import get_logger
from gtsfm.utils.vggt import VGGTReconstructionConfig, VGGTReconstructionResult

logger = get_logger()


def _resize_to_square_tensor(image: np.ndarray, target_size: int) -> torch.Tensor:
    """Resize a HxWx3 numpy image to a square torch tensor normalized to [0,1]."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return (tensor.squeeze(0)) / 255.0


def _load_vggt_inputs(loader, indices: list[int], target_size: int):
    """Load and preprocess a batch of images for VGGT."""

    def resize_transform(arr: np.ndarray) -> torch.Tensor:
        return _resize_to_square_tensor(arr, target_size)

    return loader.load_image_batch_vggt(indices, target_size, resize_transform)


def _run_vggt_pipeline(image_batch: torch.Tensor, seed: int, **kwargs) -> VGGTReconstructionResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logger.info("🔵 Running VGGT on %d images.", image_batch.shape[0])
    return vggt.run_reconstruction(image_batch, **kwargs)


def _save_reconstruction_as_text(
    result: VGGTReconstructionResult,
    results_path: Path,
    copy_to_react: bool,
    relative_results_dir: Path,
) -> None:
    target_dir = results_path / "vggt"
    target_dir.mkdir(parents=True, exist_ok=True)
    result.gtsfm_data.export_as_colmap_text(target_dir)

    if not copy_to_react:
        return

    react_destination = REACT_RESULTS_PATH / relative_results_dir / "vggt"
    react_destination.mkdir(parents=True, exist_ok=True)
    result.gtsfm_data.export_as_colmap_text(react_destination)


def _aggregate_vggt_metrics(result: VGGTReconstructionResult) -> GtsfmMetricsGroup:
    gtsfm_data = result.gtsfm_data
    num_cameras = len(gtsfm_data.get_valid_camera_indices())
    num_points3d = gtsfm_data.number_tracks()
    return GtsfmMetricsGroup(
        "vggt_runtime_metrics",
        [
            GtsfmMetric("num_cameras", num_cameras),
            GtsfmMetric("num_points3d", num_points3d),
            GtsfmMetric("used_ba", float(result.used_ba)),
        ],
    )


class ClusterVGGT(ClusterOptimizerBase):
    """Cluster optimizer that runs VGGT to generate COLMAP-style reconstructions."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        image_load_resolution: int = 1024,
        inference_resolution: int = 518,
        conf_threshold: float = 5.0,
        max_num_points: int = 100000,
        shared_camera: bool = False,
        camera_type: str = "PINHOLE",
        seed: int = 42,
        copy_results_to_react: bool = True,
        scene_dir: Optional[str] = None,
        pose_angular_error_thresh: float = 3.0,
        output_worker: Optional[str] = None,
    ) -> None:
        super().__init__(
            pose_angular_error_thresh=pose_angular_error_thresh,
            output_worker=output_worker,
        )
        self._weights_path = Path(weights_path) if weights_path is not None else None
        self._image_load_resolution = image_load_resolution
        self._inference_resolution = inference_resolution
        self._conf_threshold = conf_threshold
        self._max_points_for_colmap = max_num_points
        self._shared_camera = shared_camera
        self._camera_type = camera_type
        self._seed = seed
        self._copy_results_to_react = copy_results_to_react
        self._explicit_scene_dir = Path(scene_dir) if scene_dir is not None else None

    def __repr__(self) -> str:
        components = [
            f"weights_path={self._weights_path}",
            f"image_load_resolution={self._image_load_resolution}",
            f"inference_resolution={self._inference_resolution}",
            f"shared_camera={self._shared_camera}",
            f"camera_type={self._camera_type}",
        ]
        return "ClusterVGGT(\n  " + ",\n  ".join(str(c) for c in components) + "\n)"

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""
        # This class and its subclasses (unless overridden) are not part of the UI.
        return UiMetadata(
            display_name="VGGT",
            input_products=("Key Images",),
            output_products=("VGGT Reconstruction",),
            parent_plate="Cluster Optimizer",
        )

    def create_computation_graph(
        self,
        context: ClusterContext,
    ) -> ClusterComputationGraph | None:
        """Create the VGGT computation graph for a cluster."""

        keys = sorted(visibility_graph_keys(context.visibility_graph))
        if not keys:
            return None

        global_indices = tuple(int(idx) for idx in keys)
        image_filenames = context.loader.image_filenames()
        image_names = tuple(str(image_filenames[idx]) for idx in keys)

        config = VGGTReconstructionConfig(
            vggt_fixed_resolution=self._inference_resolution,
            img_load_resolution=self._image_load_resolution,
            confidence_threshold=self._conf_threshold,
            max_num_points=self._max_points_for_colmap,
            shared_camera=self._shared_camera,
        )

        image_batch_graph, original_coords_graph = delayed(_load_vggt_inputs, nout=2)(
            context.loader, global_indices, self._image_load_resolution
        )

        result_graph = delayed(_run_vggt_pipeline)(
            image_batch_graph,
            seed=self._seed,
            original_coords=original_coords_graph,
            image_indices=global_indices,
            image_names=image_names,
            config=config,
            weights_path=self._weights_path,
            total_num_images=context.num_images,
        )

        metrics_tasks = [delayed(_aggregate_vggt_metrics)(result_graph)]

        io_tasks: list[Delayed] = []
        with self._output_annotation():
            io_tasks.append(
                delayed(_save_reconstruction_as_text)(
                    result_graph,
                    context.output_paths.results,
                    self._copy_results_to_react,
                    context.react_results_subdir,
                )
            )

        sfm_result_graph = delayed(lambda res: res.gtsfm_data)(result_graph)

        return ClusterComputationGraph(
            io_tasks=tuple(io_tasks),
            metric_tasks=tuple(metrics_tasks),
            sfm_result=sfm_result_graph,
        )
