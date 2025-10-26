"""VGGT-based cluster optimizer leveraging the demo VGGT pipeline."""

from __future__ import annotations

import contextlib
import random
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

import numpy as np
from dask.delayed import Delayed, delayed

import gtsfm.utils.io as io_utils
from gtsfm.cluster_optimizer.cluster_optimizer_base import (
    ClusterOptimizerBase,
    REACT_RESULTS_PATH,
    logger,
)
from gtsfm.common.outputs import OutputPaths
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.visibility_graph import VisibilityGraph, visibility_graph_keys

if TYPE_CHECKING:  # pragma: no cover
    import pycolmap
    import torch


def _extract_summary(result: Tuple[dict[str, Any], GtsfmMetricsGroup]) -> dict[str, Any]:
    return result[0]


def _extract_metrics(result: Tuple[dict[str, Any], GtsfmMetricsGroup]) -> GtsfmMetricsGroup:
    return result[1]


def _ensure_dependency_paths() -> Tuple[Path, Path]:
    """Configure sys.path to locate VGGT and LightGlue third-party dependencies."""
    repo_root = Path(__file__).resolve().parents[2]
    vggt_root = repo_root / "thirdparty" / "vggt"
    lightglue_root = repo_root / "thirdparty" / "LightGlue"
    for path in (vggt_root, lightglue_root):
        if path.exists():
            str_path = str(path)
            if str_path not in sys.path:
                sys.path.insert(0, str_path)
    return vggt_root, lightglue_root


def _build_pycolmap_intri(
    fidx: int, intrinsics: np.ndarray, camera_type: str, extra_params: Optional[np.ndarray] = None
) -> np.ndarray:
    """Construct camera intrinsics vector for pycolmap."""
    if camera_type == "PINHOLE":
        return np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    if camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        return np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    if camera_type == "SIMPLE_RADIAL":
        if extra_params is None:
            raise ValueError("Extra parameters required for SIMPLE_RADIAL cameras.")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        return np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    raise ValueError(f"Camera type {camera_type} is not supported.")


def batch_np_matrix_to_pycolmap_wo_track(
    points3d: np.ndarray,
    points_xyf: np.ndarray,
    points_rgb: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_size: np.ndarray,
    shared_camera: bool = False,
    camera_type: str = "SIMPLE_PINHOLE",
) -> "pycolmap.Reconstruction":
    """Convert batched numpy arrays produced by VGGT into a pycolmap reconstruction."""
    import pycolmap

    reconstruction = pycolmap.Reconstruction()
    num_frames = len(extrinsics)
    num_points = len(points3d)

    for vidx in range(num_points):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    for fidx in range(num_frames):
        if camera is None or not shared_camera:
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)
            camera = pycolmap.Camera(
                model=camera_type,
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=pycolmap_intri,
                camera_id=fidx + 1,
            )
            reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3])
        image = pycolmap.Image(
            id=fidx + 1,
            name=f"image_{fidx + 1}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        points2d_list = []
        point2d_idx = 0

        mask = points_xyf[:, 2].astype(np.int32) == fidx
        points_belonging = np.nonzero(mask)[0]

        for point3d_batch_idx in points_belonging:
            point3d_id = point3d_batch_idx + 1
            point2d_xyf = points_xyf[point3d_batch_idx]
            point2d_xy = point2d_xyf[:2]
            points2d_list.append(pycolmap.Point2D(point2d_xy, point3d_id))

            track = reconstruction.points3D[point3d_id].track
            track.add_element(fidx + 1, point2d_idx)
            point2d_idx += 1

        if points2d_list:
            try:
                image.points2D = pycolmap.ListPoint2D(points2d_list)
            except RuntimeError:
                logger.warning("Frame %d does not have any valid points for pycolmap.", fidx + 1)

        reconstruction.add_image(image)

    return reconstruction


def rename_colmap_recons_and_rescale_camera(
    reconstruction: "pycolmap.Reconstruction",
    image_paths: Sequence[str],
    original_coords: np.ndarray,
    img_size: int,
    shift_point2d_to_original_res: bool = False,
    shared_camera: bool = False,
) -> "pycolmap.Reconstruction":
    """Rescale COLMAP reconstruction to match original image resolution and restore filenames."""
    import pycolmap

    rescale_camera = True
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp

            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2d in pyimage.points2D:
                point2d.xy = (point2d.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


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

    @classmethod
    def simple(
        cls,
        weights_path: Optional[str] = None,
        use_ba: bool = False,
        inference_resolution: int = 518,
        image_load_resolution: int = 1024,
        output_worker: Optional[str] = None,
    ) -> "ClusterVGGT":
        """Lightweight constructor with sensible defaults for quick instantiation.

        Args:
            weights_path: optional path to VGGT weights; if None the class will look for default weights in
                the thirdparty/vggt/weights directory at runtime.
            use_ba: whether to run BA after VGGT (defaults to False — feed-forward mode).
            inference_resolution: resolution used during VGGT inference (default 518).
            image_load_resolution: resolution used when loading full images (default 1024).
            output_worker: optional Dask worker string for routing I/O tasks.
        """
        return cls(
            weights_path=weights_path,
            image_load_resolution=image_load_resolution,
            inference_resolution=inference_resolution,
            use_ba=use_ba,
            output_worker=output_worker,
        )

    def create_computation_graph(
        self,
        num_images: int,
        one_view_data_dict: dict[int, OneViewData],
        output_paths: OutputPaths,
        loader: LoaderBase,
        output_root: Path,
        visibility_graph: VisibilityGraph,
        image_futures: list[Any],
    ) -> Optional[Tuple[Delayed, Sequence[Delayed], Sequence[Delayed]]]:
        del num_images, one_view_data_dict, output_root, image_futures  # Unused in VGGT pipeline.

        image_indices = self._collect_image_indices(visibility_graph, len(loader))
        if not image_indices:
            logger.warning("ClusterVGGT: no images available for VGGT inference; skipping cluster.")
            return None

        scene_dir = self._explicit_scene_dir or (output_paths.results / "vggt")

        with self._output_annotation():
            prepare_graph = delayed(self._prepare_scene_images)(loader, image_indices, scene_dir)

        run_graph = delayed(self._run_vggt_pipeline)(scene_dir, prepare_graph)

        summary_graph = delayed(_extract_summary)(run_graph)
        metrics_graph = delayed(_extract_metrics)(run_graph)
        metrics_list = [metrics_graph]

        delayed_results: list[Delayed] = []
        if self._copy_results_to_react:
            with self._output_annotation():
                delayed_results.append(delayed(self._copy_scene_to_react)(scene_dir, summary_graph))

        return summary_graph, delayed_results, metrics_list

    @staticmethod
    def _collect_image_indices(visibility_graph: VisibilityGraph, num_images: int) -> Sequence[int]:
        if visibility_graph:
            indices = sorted(visibility_graph_keys(visibility_graph))
            return indices
        return list(range(num_images))

    def _prepare_scene_images(
        self,
        loader: LoaderBase,
        image_indices: Sequence[int],
        scene_dir: Path,
    ) -> Sequence[str]:
        images_dir = scene_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)

        filenames = loader.image_filenames() if hasattr(loader, "image_filenames") else None
        saved_paths: list[str] = []
        for idx in image_indices:
            image = loader.get_image_full_res(idx)
            file_name = image.file_name or (filenames[idx] if filenames and idx < len(filenames) else None)
            if file_name is None:
                file_name = f"image_{idx:03d}.png"
            save_path = images_dir / file_name
            io_utils.save_image(image, str(save_path))
            saved_paths.append(str(save_path))
        return saved_paths

    def _run_vggt_pipeline(
        self, scene_dir: Path, image_paths: Sequence[str]
    ) -> Tuple[dict[str, Any], GtsfmMetricsGroup]:
        if self._use_ba:
            raise NotImplementedError("ClusterVGGT currently supports feed-forward VGGT (use_ba=False) only.")

        start_time = time.time()
        vggt_root, _ = _ensure_dependency_paths()

        import torch
        import torch.nn.functional as F
        import pycolmap
        import trimesh
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            dtype = torch.float32

        weights_path = self._weights_path or (vggt_root / "weights" / "model.pt")
        if not weights_path.exists():
            raise FileNotFoundError(f"VGGT weights not found at {weights_path}.")

        model = VGGT()
        checkpoint = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        model.eval()
        model = model.to(device)

        images, original_coords = load_and_preprocess_images_square(image_paths, self._image_load_resolution)
        images = images.to(device)
        original_coords = original_coords.to(device)

        extrinsic, intrinsic, depth_map, depth_conf = self._forward_vggt(model, images, dtype, device)
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

        image_size = np.array([self._inference_resolution, self._inference_resolution])
        points_rgb = F.interpolate(
            images, size=(self._inference_resolution, self._inference_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        num_frames, height, width, _ = points_3d.shape
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= self._conf_threshold
        conf_mask = randomly_limit_trues(conf_mask, self._max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        if len(points_3d) == 0:
            raise RuntimeError("VGGT feed-forward pipeline produced no valid 3D points.")

        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d=points_3d,
            points_xyf=points_xyf,
            points_rgb=points_rgb,
            extrinsics=extrinsic,
            intrinsics=intrinsic,
            image_size=image_size,
            shared_camera=self._shared_camera,
            camera_type=self._camera_type,
        )

        reconstruction = rename_colmap_recons_and_rescale_camera(
            reconstruction,
            [Path(path).name for path in image_paths],
            original_coords.cpu().numpy(),
            img_size=self._inference_resolution,
            shift_point2d_to_original_res=True,
            shared_camera=self._shared_camera,
        )

        sparse_dir = scene_dir / ("sparse_w_ba" if self._use_ba else "sparse_wo_ba")
        sparse_dir.mkdir(parents=True, exist_ok=True)
        reconstruction.write(str(sparse_dir))
        trimesh.PointCloud(points_3d, colors=points_rgb).export(sparse_dir / "points.ply")

        duration = time.time() - start_time
        metrics = GtsfmMetricsGroup(
            "vggt_runtime_metrics",
            [
                GtsfmMetric("total_runtime_sec", float(duration)),
                GtsfmMetric("num_input_images", len(image_paths)),
            ],
        )
        summary = {
            "reconstruction_dir": str(sparse_dir),
            "use_ba": self._use_ba,
            "num_input_images": len(image_paths),
        }
        return summary, metrics

    def _forward_vggt(
        self,
        model: Any,
        images: "torch.Tensor",
        dtype: "torch.dtype",
        device: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import torch
        import torch.nn.functional as F
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        resized_images = F.interpolate(
            images, size=(self._inference_resolution, self._inference_resolution), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            if device == "cuda":
                autocast_cm = torch.cuda.amp.autocast(dtype=dtype)
            else:
                autocast_cm = contextlib.nullcontext()
            with autocast_cm:
                images_batch = resized_images[None]
                aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)

        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()
        return extrinsic, intrinsic, depth_map, depth_conf

    @staticmethod
    def _copy_scene_to_react(scene_dir: Path, summary: dict[str, Any]) -> None:
        del summary  # Value carried to enforce dependency.
        destination = REACT_RESULTS_PATH / scene_dir.name
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(scene_dir, destination)
