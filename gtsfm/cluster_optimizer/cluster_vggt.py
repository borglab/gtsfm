"""VGGT-based cluster optimizer leveraging the demo VGGT pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterOptimizerBase


class ClusterVGGT(ClusterOptimizerBase):
    """Cluster optimizer that runs VGGT to generate COLMAP-style reconstructions."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        image_load_resolution: int = 1024,
        inference_resolution: int = 518,
        conf_threshold: float = 5.0,
        max_points_for_colmap: int = 100000,
        use_ba: bool = False,
        shared_camera: bool = False,
        camera_type: str = "PINHOLE",
        seed: int = 42,
        copy_results_to_react: bool = True,
        scene_dir: Optional[str] = None,
        pose_angular_error_thresh: float = 3.0,
        output_worker: Optional[str] = None,
    ) -> None:
        super().__init__(
            correspondence_generator=None,
            pose_angular_error_thresh=pose_angular_error_thresh,
            output_worker=output_worker,
        )
        self._weights_path = Path(weights_path) if weights_path is not None else None
        self._image_load_resolution = image_load_resolution
        self._inference_resolution = inference_resolution
        self._conf_threshold = conf_threshold
        self._max_points_for_colmap = max_points_for_colmap
        self._use_ba = use_ba
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
            f"use_ba={self._use_ba}",
            f"shared_camera={self._shared_camera}",
            f"camera_type={self._camera_type}",
        ]
        return "ClusterVGGT(\n  " + ",\n  ".join(str(c) for c in components) + "\n)"
