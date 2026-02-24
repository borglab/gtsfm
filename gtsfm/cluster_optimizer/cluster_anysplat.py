"""Feed forward Gaussian Splatting using the AnySplat model.
https://github.com/InternRobotics/AnySplat
Authors: Harneet Singh Khanuja
"""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Hashable, List

import cv2
import gtsam
import numpy as np
import torch
import torchvision  # type: ignore
from dask import delayed  # type: ignore
from dask.delayed import Delayed

import gtsfm.frontend.anysplat as anysplat_utils
import gtsfm.utils.torch as torch_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterComputationGraph, ClusterContext, ClusterOptimizerBase
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.anysplat import AnySplatReconstructionResult
from gtsfm.frontend.anysplat import (
    batchify_unproject_depth_map_to_point_map as _anysplat_batchify_unproject,
)  # type: ignore
from gtsfm.ui.gtsfm_process import UiMetadata
from gtsfm.utils import align as align_utils
from gtsfm.utils import logger as logger_utils
from gtsfm.utils.transform import transform_gaussian_splats

logger = logger_utils.get_logger()

# Module-level cache to reuse AnySplat weights per worker process.
_MODEL_CACHE: dict[Hashable, Any] = {}


def save_splats(result: GtsfmData, save_gs_files_path: Path) -> None:
    splats = result.get_gaussian_splats()
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


def _save_reconstruction_as_text(
    result: GtsfmData,
    results_path: Path,
) -> None:
    target_dir = results_path / "anysplat"
    target_dir.mkdir(parents=True, exist_ok=True)
    result.export_as_colmap_text(target_dir)
    save_splats(result, target_dir)


class ClusterAnySplat(ClusterOptimizerBase):
    """Class for AnySplat (feed forward GS implementation)"""

    def __init__(
        self,
        model_loader: Callable[[], Any] | None = None,
        *,
        local_checkpoint: str | Path | None = None,
        model_cache_key: Hashable | None = None,
        max_num_points: int | None = None,
        tracking: bool | None = True,
        max_query_pts: int | None = 2048,
        query_frame_num: int | None = 5,
        keypoint_extractor: str | None = "aliked+sp",
        max_points_num: int | None = 163840,
        fine_tracking: bool | None = True,
        track_vis_thresh: float | None = 0.9,
        num_inliers: int | None = 3,
        confidence_thresh: float | None = 0.0,
        reproj_error_thresh: float | None = None,
        run_bundle_adjustment_on_leaf: bool | None = False,
        run_bundle_adjustment_on_parent: bool | None = True,
        plot_reprojection_histograms: bool | None = True,
    ):
        """
        Initializes the ClusterAnySplat optimizer.
        Args:
            model_loader (Callable[[], Any] | None): Optional custom model loader function.
            local_checkpoint (str | Path | None): Local filesystem path to the model weights checkpoint.
            model_cache_key (Hashable | None): Key used to cache/reuse the loaded model across worker processes.
            max_num_points (int | None): Maximum number of points (gaussian means) to save if tracking is False
            tracking (bool | None): Boolean used for signaling if tracking will be used for merging
            max_query_pts (int | None): VGGT tracking argument with its default value
            query_frame_num (int | None): VGGT tracking argument with its default value
            keypoint_extractor (str | None): VGGT tracking argument with its default value
            max_points_num (int | None): VGGT tracking argument with its default value
            fine_tracking (int | None): VGGT tracking argument with its default value
            track_vis_thresh (int | None): VGGT tracking argument with its default value
            num_inliers (int | None): threshold for considering a track valid
            confidence_thresh (float | None): minimum confidence required for a retained track
            reproj_error_thresh (float | None): optional per-track reprojection error ceiling in pixels
            run_bundle_adjustment_on_leaf (bool | None): optional BA operation on individual cluster (Default: False)
            run_bundle_adjustment_on_parent (bool | None): optional BA operation on after camera merging (Default: True)
            plot_reprojection_histograms (bool | None): optional plotting reprojection error histograms (Default: True)
        """
        super().__init__()
        self._model = None
        self._device = torch_utils.default_device()
        self.max_gaussians = max_num_points
        self.tracking = tracking
        self.max_query_pts = max_query_pts
        self.query_frame_num = query_frame_num
        self.max_points_num = max_points_num
        self.keypoint_extractor = keypoint_extractor
        self.fine_tracking = fine_tracking
        self.track_vis_thresh = track_vis_thresh
        self.num_inliers = num_inliers
        self.confidence_thresh = confidence_thresh
        self.reproj_error_thresh = reproj_error_thresh
        self.run_bundle_adjustment_on_leaf = run_bundle_adjustment_on_leaf
        self.run_bundle_adjustment_on_parent = run_bundle_adjustment_on_parent
        self.plot_reprojection_histograms = plot_reprojection_histograms

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
        voxel_count = result.gtsfm_data.get_gaussian_splats().means.shape[1]
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

        depth_outputs: dict[str, torch.Tensor] = {}
        depth_head = getattr(getattr(self._model, "encoder", None), "depth_head", None)

        def _capture_depth(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            outputs: Any,
        ) -> None:
            if depth_outputs or not isinstance(outputs, tuple) or len(outputs) < 2:
                return
            depth, confidence = outputs[:2]
            depth_outputs["depth_map"] = depth.detach().cpu()
            depth_outputs["depth_confidence"] = confidence.detach().cpu()

        hook = depth_head.register_forward_hook(_capture_depth) if depth_head else None

        try:
            splats, pred_context_pose = self._model.inference((processed_images + 1) * 0.5)
        finally:
            if hook:
                hook.remove()
        depth_map = depth_outputs.get("depth_map")
        depth_confidence = depth_outputs.get("depth_confidence")

        # Move results to CPU to avoid Dask serialization errors.
        for attr in ["means", "scales", "rotations", "harmonics", "opacities", "covariances"]:
            setattr(splats, attr, getattr(splats, attr).cpu())
        pred_context_pose["extrinsic"] = pred_context_pose["extrinsic"].cpu()  # anysplat returns w2c extrinsics
        pred_context_pose["intrinsic"] = pred_context_pose[
            "intrinsic"
        ].cpu()  # the intrinsics received from anysplat are normalized
        decoder = self._model.decoder.cpu()

        gtsfm_data = GtsfmData(number_images=len(images))
        for local_idx, (global_idx, img) in enumerate(images.items()):
            intrinsic = pred_context_pose["intrinsic"][0][local_idx].numpy()
            extrinsic = pred_context_pose["extrinsic"][0][local_idx].numpy()
            intrinsics_pixels = intrinsic.copy()
            intrinsics_pixels[..., 0, :] *= width
            intrinsics_pixels[..., 1, :] *= height
            # TODO(akshay-krishnan): Add support for pinhole camera model.
            # TODO(akshay-krishnan): Fix crop coords.
            crop_coords = np.array([0, 0, width, height])
            camera = torch_utils.camera_from_matrices(
                extrinsic, intrinsics_pixels, crop_coords=crop_coords, wTc_flag=True, use_cal3_bundler=True
            )
            gtsfm_data.add_camera(global_idx, camera)  # type: ignore
            gtsfm_data.set_image_info(
                global_idx,
                name=img.file_name,
                shape=(height, width),  # while the image while passing through the model inference in 448,448
                # the intrinsic outputs were initially normalized so unnormalized ones are used to create gtsfm data
            )

        if not self.tracking:
            logger.info("Tracking Disabled, will use camera-only merging")
            gtsfm_data = anysplat_utils.add_tracks_with_gaussian_mean(
                splats,
                self.max_gaussians,
                gtsfm_data,
            )
            del depth_map, depth_confidence

        gtsfm_data.set_gaussian_splats(splats)

        if self.tracking:
            logger.info("Will use track correspondences in merging")
            dense_points = None
            extrinsic_wTc = pred_context_pose["extrinsic"]
            extrinsic_cTw = torch.linalg.inv(extrinsic_wTc)[..., :3, :]
            intrinsics_norm = pred_context_pose["intrinsic"]
            intrinsics_pixels = intrinsics_norm.clone()
            intrinsics_pixels[..., 0, :] *= width
            intrinsics_pixels[..., 1, :] *= height
            # the depth map and depth confidence are 448,448 shapes so normalized intrinsics are unnormalized
            dense_points = _anysplat_batchify_unproject(depth_map, extrinsic_cTw, intrinsics_pixels)

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "VGGT tracking requires a CUDA-capable GPU (DINO uses flash attention). "
                    "Re-run the pipeline with a CUDA GPU available."
                )
            device = torch.device("cuda")
            dense_points = dense_points.to(device)
            dtype = torch.float32  # Tracker stack (LightGlue / DINO) expects fp32 inputs.

            if processed_images.device != device or processed_images.dtype != dtype:
                logger.info("Moving VGGT tracking inputs to %s (dtype=%s) for DINO attention.", device, dtype)
                tracking_images = processed_images.to(device=device, dtype=dtype, non_blocking=True)
                tracking_images = ((tracking_images + 1) / 2).clamp(0, 1)
            else:
                tracking_images = ((processed_images + 1) / 2).clamp(0, 1)

            conf_tensor = depth_confidence.squeeze(0).to(device="cpu")
            points_tensor = dense_points.squeeze(0).to(device="cpu")
            predict_tracks = anysplat_utils.import_predict_tracks()
            with torch.no_grad():
                tracks, vis_scores, confidences, points_3d, colors = predict_tracks(
                    tracking_images.squeeze(0),
                    conf=conf_tensor,
                    points_3d=points_tensor,
                    masks=None,  # ignored anyway !
                    max_query_pts=self.max_query_pts,
                    query_frame_num=self.query_frame_num,
                    keypoint_extractor=self.keypoint_extractor,
                    max_points_num=self.max_points_num,
                    fine_tracking=self.fine_tracking,
                )

            del (
                colors,
                dense_points,
                depth_map,
                depth_confidence,
                conf_tensor,
                points_tensor,
                tracking_images,
            )

            logger.info("🔢 Number of tracks before filtering %s", points_3d.shape[0])
            track_mask = vis_scores > self.track_vis_thresh
            inlier_num = track_mask.sum(0)
            self.num_inliers = min(len(images), 3)
            valid_mask = inlier_num >= self.num_inliers  # a track is invalid if without mentioned number of inliers

            logger.info("🔢 Valid track count after inlier filtering %s", sum(valid_mask))

            confidence_mask = confidences >= self.confidence_thresh
            valid_mask = np.logical_and(valid_mask, confidence_mask)

            logger.info("🔢 Valid track count after confidence filtering %s", sum(valid_mask))

            if self.reproj_error_thresh is not None:
                reprojection_errors = anysplat_utils.compute_reprojection_errors(
                    points_3d=points_3d,
                    tracks_2d=tracks,
                    track_mask=track_mask,
                    extrinsic_cTw=extrinsic_cTw,
                    intrinsics_pixels=intrinsics_pixels,
                )
                finite_reproj_mask = np.isfinite(reprojection_errors)
                # if np.any(finite_reproj_mask):
                #     anysplat_utils.log_reprojection_metrics_per_track(reprojection_errors, finite_reproj_mask)
                # else:
                #     logger.info("No valid reprojection errors could be computed for the current cluster.")

                reproj_mask = np.logical_and(finite_reproj_mask, reprojection_errors < self.reproj_error_thresh)
                valid_mask = np.logical_and(valid_mask, reproj_mask)

                logger.info("Valid track count after reprojection error filtering %s", sum(valid_mask))

                # anysplat_utils.log_reprojection_metrics_per_track(reprojection_errors, valid_mask)

                del reproj_mask, finite_reproj_mask

            del confidences, vis_scores

            valid_idx = np.nonzero(valid_mask)[0]

            global_indices = list(images.keys())
            for valid_id in valid_idx:
                track = torch_utils.colored_track_from_point(points_3d[valid_id], np.zeros(3).astype(float).tolist())
                frame_idx = np.where(track_mask[:, valid_id])[0]
                for local_id in frame_idx:
                    global_idx = global_indices[local_id]
                    u, v = tracks[local_id, valid_id]

                    track.addMeasurement(global_idx, gtsam.Point2(u, v))  # if we save normalized tracks,
                    # then we would have to normalize u and v as well
                gtsfm_data.add_track(track)

            logger.info("📏 Reprojection error stats after filtering")
            gtsfm_data.log_scene_reprojection_error_stats()

        if self.run_bundle_adjustment_on_leaf:
            if gtsfm_data.number_tracks() == 0:
                logger.warning("Skipping bundle adjustment because VGGT produced no valid tracks.")
            else:
                try:
                    # TODO(akshay-krishnan): Configure this to be same as VGGT's bundle adjustment optimizer.
                    post_ba_gtsfm_data, _ = BundleAdjustmentOptimizer().run_simple_ba(gtsfm_data)
                    for idx in post_ba_gtsfm_data.get_valid_camera_indices():
                        info = gtsfm_data.get_image_info(idx)
                        post_ba_gtsfm_data.set_image_info(idx, name=info.name, shape=info.shape)
                    postba_S_preba = align_utils.sim3_from_Pose3_maps(post_ba_gtsfm_data.poses(), gtsfm_data.poses())
                    post_ba_gaussians = transform_gaussian_splats(
                        gtsfm_data.get_gaussian_splats(), postba_S_preba  # type: ignore
                    )
                    post_ba_gtsfm_data.set_gaussian_splats(post_ba_gaussians)

                    logger.info("📏 Reprojection error stats after running BA on individual node")
                    post_ba_gtsfm_data.log_scene_reprojection_error_stats()
                    return AnySplatReconstructionResult(
                        post_ba_gtsfm_data,
                        splats,
                        pred_context_pose,
                        height,
                        width,
                        decoder,
                    )
                except Exception as exc:
                    logger.warning("⚠️ Failed to run bundle adjustment: %s", exc)

        return AnySplatReconstructionResult(
            gtsfm_data,
            splats,
            pred_context_pose,
            height,
            width,
            decoder,
        )

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
        splats = result.gtsfm_data.get_gaussian_splats()
        for attr in ["means", "scales", "rotations", "harmonics", "opacities", "covariances"]:
            setattr(splats, attr, getattr(splats, attr).to(device))

        b = 1  # AnySplat convention
        # anysplat function internally scales the normalized intrinsics to height and width
        anysplat_utils.save_interpolated_video(
            extrinsics,
            intrinsics,
            b,
            result.height,
            result.width,
            splats,
            save_gs_files_path,
            decoder_model,
        )

        del intrinsics, extrinsics, decoder_model
        for attr in ["means", "scales", "rotations", "harmonics", "opacities", "covariances"]:
            setattr(splats, attr, getattr(splats, attr).cpu())

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
                delayed(_save_reconstruction_as_text)(
                    result_graph.gtsfm_data,
                    context.output_paths.results,
                )
            )
            io_tasks.append(
                delayed(self._generate_interpolated_video)(
                    result_graph,
                    str(context.output_paths.results / "anysplat"),
                )
            )

        sfm_result_graph = delayed(lambda res: res.gtsfm_data)(result_graph)

        return ClusterComputationGraph(
            io_tasks=tuple(io_tasks),
            metric_tasks=tuple(metrics_tasks),
            sfm_result=sfm_result_graph,
        )
