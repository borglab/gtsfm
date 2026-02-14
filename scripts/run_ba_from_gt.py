"""Run bundle adjustment initialized from ground-truth cameras and pre-BA tracks.

This script:
1) Loads pre-BA data (COLMAP model).
2) Filters pre-BA tracks by reprojection error.
3) Transfers only 2D measurements onto a GT-initialized scene.
4) Triangulates 3D points from those 2D measurements.
5) Runs bundle adjustment starting from the GT-initialized scene.
6) Computes pose AUCs and reprojection error plots for GT-init vs post-BA.

Example:
    python scripts/run_ba_from_gt.py \
        --pre_ba_dir /path/to/vggt_pre_ba \
        --output_dir /path/to/vggt_post_ba \
        --dataset_dir ../data/gerrard-hall \
        --loader colmap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from gtsam import Pose3, SfmTrack
import gtsam

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.common import types as gtsfm_types
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationOptions,
    TriangulationSamplingMode,
)
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.loader.olsson_loader import OlssonLoader
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.utils import reprojection as reprojection_utils
from gtsfm.visualization.track_viz_utils import visualize_reprojection_overlays

logger = logger_utils.get_logger()

LOADER_CHOICES = ("colmap", "olsson")


def _create_loader(
    loader_type: str,
    dataset_dir: str,
    images_dir: Optional[str],
    use_gt_intrinsics: bool,
    use_gt_extrinsics: bool,
    max_resolution: int,
    max_frame_lookahead: int,
) -> ColmapLoader | OlssonLoader:
    if loader_type == "colmap":
        return ColmapLoader(
            dataset_dir=dataset_dir,
            images_dir=images_dir,
            use_gt_intrinsics=use_gt_intrinsics,
            use_gt_extrinsics=use_gt_extrinsics,
            max_resolution=max_resolution,
        )
    if loader_type == "olsson":
        return OlssonLoader(
            dataset_dir=dataset_dir,
            images_dir=images_dir,
            use_gt_intrinsics=use_gt_intrinsics,
            use_gt_extrinsics=use_gt_extrinsics,
            max_frame_lookahead=max_frame_lookahead,
            max_resolution=max_resolution,
        )
    raise ValueError(f"Unknown loader type: {loader_type}")


def _load_pre_ba_data(pre_ba_dir: str) -> GtsfmData:
    pre_ba_path = Path(pre_ba_dir)
    if not pre_ba_path.exists():
        raise FileNotFoundError(f"Pre-BA directory not found: {pre_ba_dir}")
    return GtsfmData.read_colmap(str(pre_ba_path))


def _save_post_ba_data(post_ba_data: GtsfmData, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    post_ba_data.export_as_colmap_text(str(output_path))


def _get_pose_metrics(
    result_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: Optional[list[str]] = None,
    save_dir: Optional[str] = None,
) -> GtsfmMetricsGroup:
    """Compute pose metrics for a result after aligning with ground truth."""
    poses_gt: dict[int, Pose3] = {}
    if gt_filenames:
        name_to_idx = {name: idx for idx, name in enumerate(gt_filenames)}
        for i in result_data.get_valid_camera_indices():
            name = result_data.get_image_info(i).name
            if name is None:
                continue
            gt_idx = name_to_idx.get(name) or name_to_idx.get(Path(name).name)
            if gt_idx is None or gt_idx >= len(cameras_gt):
                continue
            camera = cameras_gt[gt_idx]
            if camera is not None:
                poses_gt[i] = camera.pose()
    else:
        image_idxs = list(result_data.cameras().keys())
        for i in image_idxs:
            if i >= len(cameras_gt):
                continue
            camera = cameras_gt[i]
            if camera is not None:
                poses_gt[i] = camera.pose()
    if len(poses_gt) == 0:
        return GtsfmMetricsGroup(name="ba_pose_error_metrics", metrics=[])
    aligned_result_data = result_data.align_via_sim3_and_transform(poses_gt)
    computed_wTi: dict[int, Optional[Pose3]] = {i: pose for i, pose in aligned_result_data.get_camera_poses().items()}
    return metrics_utils.compute_ba_pose_metrics(
        gt_wTi=poses_gt,
        computed_wTi=computed_wTi,
        save_dir=save_dir,
        store_full_data=True,
    )


def _build_gt_init_data(
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: Optional[list[str]],
    gt_shapes: Optional[list[Optional[tuple[int, int]]]],
    triangulated_tracks: list[SfmTrack],
) -> GtsfmData:
    data = GtsfmData(number_images=len(cameras_gt))
    for idx, camera in enumerate(cameras_gt):
        if camera is not None:
            data.add_camera(idx, camera)
        if gt_filenames is not None and idx < len(gt_filenames):
            shape = gt_shapes[idx] if gt_shapes is not None and idx < len(gt_shapes) else None
            data.set_image_info(idx, name=gt_filenames[idx], shape=shape)
    for track in triangulated_tracks:
        data.add_track(track)
    return data


def _normalize_shape(shape: Optional[tuple[int, ...]]) -> Optional[tuple[int, int]]:
    if shape is None or len(shape) == 0:
        return None
    height = int(shape[0])
    width = int(shape[1]) if len(shape) > 1 else height
    if height <= 0 or width <= 0:
        return None
    return (height, width)


def _fallback_shape_from_camera(camera: Optional[gtsfm_types.CAMERA_TYPE]) -> Optional[tuple[int, int]]:
    if camera is None:
        return None
    calibration = camera.calibration()
    cx = getattr(calibration, "px", None)
    cy = getattr(calibration, "py", None)
    if callable(cx):
        cx = cx()
    if callable(cy):
        cy = cy()
    if isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
        width = max(int(round(cx * 2)), 1)
        height = max(int(round(cy * 2)), 1)
        return (height, width)
    return None


def _resolve_shape(
    shape: Optional[tuple[int, ...]], camera: Optional[gtsfm_types.CAMERA_TYPE]
) -> Optional[tuple[int, int]]:
    normalized = _normalize_shape(shape)
    if normalized is not None:
        return normalized
    return _fallback_shape_from_camera(camera)


def _filter_gt_to_preba(
    pre_ba_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: list[str],
    gt_shapes: Optional[list[Optional[tuple[int, int]]]],
) -> tuple[
    list[Optional[gtsfm_types.CAMERA_TYPE]],
    list[str],
    Optional[list[Optional[tuple[int, int]]]],
]:
    preba_names: set[str] = set()
    for cam_idx in pre_ba_data.get_valid_camera_indices():
        name = pre_ba_data.get_image_info(cam_idx).name
        if name is None:
            continue
        preba_names.add(name)
        preba_names.add(Path(name).name)

    filtered_cameras: list[Optional[gtsfm_types.CAMERA_TYPE]] = []
    filtered_filenames: list[str] = []
    filtered_shapes: list[Optional[tuple[int, int]]] = []

    for idx, name in enumerate(gt_filenames):
        if name not in preba_names and Path(name).name not in preba_names:
            continue
        filtered_filenames.append(name)
        filtered_cameras.append(cameras_gt[idx] if idx < len(cameras_gt) else None)
        if gt_shapes is not None and idx < len(gt_shapes):
            filtered_shapes.append(gt_shapes[idx])
        elif gt_shapes is not None:
            filtered_shapes.append(None)

    dropped = len(gt_filenames) - len(filtered_filenames)
    if dropped > 0:
        logger.info("Discarded %d GT cameras not present in pre-BA data.", dropped)
    return (
        filtered_cameras,
        filtered_filenames,
        filtered_shapes if gt_shapes is not None else None,
    )


def _override_gt_intrinsics_with_preba(
    pre_ba_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: list[str],
) -> list[Optional[gtsfm_types.CAMERA_TYPE]]:
    preba_name_to_camera: dict[str, gtsfm_types.CAMERA_TYPE] = {}
    for cam_idx in pre_ba_data.get_valid_camera_indices():
        name = pre_ba_data.get_image_info(cam_idx).name
        if name is None:
            continue
        cam = pre_ba_data.get_camera(cam_idx)
        if cam is None:
            continue
        preba_name_to_camera[name] = cam
        preba_name_to_camera[Path(name).name] = cam

    updated: list[Optional[gtsfm_types.CAMERA_TYPE]] = []
    for idx, gt_cam in enumerate(cameras_gt):
        if gt_cam is None:
            updated.append(None)
            continue
        name = gt_filenames[idx]
        preba_cam = preba_name_to_camera.get(name) or preba_name_to_camera.get(Path(name).name)
        if preba_cam is None:
            updated.append(gt_cam)
            continue
        calib = preba_cam.calibration()
        cam_class = gtsfm_types.get_camera_class_for_calibration(calib)
        f = (calib.fx() + calib.fy()) / 2.0
        new_calib = gtsam.Cal3_S2(f, f, 0.0, calib.px(), calib.py())
        # new_calib = gtsam.Cal3DS2(f, f, 0.0, calib.px(), calib.py(), 0.0, 0.0, 0.0, 0.0)
        print("cam_class ", idx, type(cam_class), type(gt_cam.calibration()), gt_cam.calibration(), calib)
        updated.append(gtsam.PinholeCameraCal3_S2(gt_cam.pose(), new_calib))  # type: ignore
    return updated


def _log_intrinsics_comparison(
    pre_ba_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: list[str],
) -> None:
    def _extract_params(calib: object) -> dict[str, float]:
        params: dict[str, float] = {}
        for name in ("fx", "fy", "px", "py", "skew", "k1", "k2", "k3", "k4", "p1", "p2"):
            attr = getattr(calib, name, None)
            if attr is None:
                continue
            try:
                value = attr() if callable(attr) else attr
            except Exception:
                continue
            if isinstance(value, (int, float)):
                params[name] = float(value)
        return params

    preba_name_to_camera: dict[str, gtsfm_types.CAMERA_TYPE] = {}
    for cam_idx in pre_ba_data.get_valid_camera_indices():
        name = pre_ba_data.get_image_info(cam_idx).name
        if name is None:
            continue
        cam = pre_ba_data.get_camera(cam_idx)
        if cam is None:
            continue
        preba_name_to_camera[name] = cam
        preba_name_to_camera[Path(name).name] = cam

    logger.info("GT vs pre-BA intrinsics (per camera):")
    for idx, gt_cam in enumerate(cameras_gt):
        if gt_cam is None:
            continue
        name = gt_filenames[idx]
        preba_cam = preba_name_to_camera.get(name) or preba_name_to_camera.get(Path(name).name)
        if preba_cam is None:
            continue
        gt_calib = gt_cam.calibration()
        pre_calib = preba_cam.calibration()
        gt_params = _extract_params(gt_calib)
        pre_params = _extract_params(pre_calib)
        all_keys = sorted(set(gt_params) | set(pre_params))
        diffs = {key: pre_params.get(key, float("nan")) - gt_params.get(key, float("nan")) for key in all_keys}
        logger.info(
            "  %s gt=%s pre_ba=%s diff=%s",
            name,
            gt_params,
            pre_params,
            diffs,
        )


def _log_image_scales(
    pre_ba_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: list[str],
    gt_shapes: Optional[list[Optional[tuple[int, int]]]],
) -> None:
    name_to_gt_idx: dict[str, int] = {name: idx for idx, name in enumerate(gt_filenames)}
    for name, idx in list(name_to_gt_idx.items()):
        name_to_gt_idx[Path(name).name] = idx

    gt_shapes_by_idx: dict[int, Optional[tuple[int, int]]] = {}
    if gt_shapes is not None:
        gt_shapes_by_idx = {idx: _normalize_shape(shape) for idx, shape in enumerate(gt_shapes)}

    logger.info("Image scale factors (pre-BA -> GT):")
    for pre_idx in pre_ba_data.get_valid_camera_indices():
        pre_name = pre_ba_data.get_image_info(pre_idx).name
        if pre_name is None:
            continue
        gt_idx = name_to_gt_idx.get(pre_name) or name_to_gt_idx.get(Path(pre_name).name)
        if gt_idx is None or gt_idx >= len(cameras_gt):
            continue
        pre_shape = _resolve_shape(pre_ba_data.get_image_info(pre_idx).shape, pre_ba_data.get_camera(pre_idx))
        gt_shape = gt_shapes_by_idx.get(gt_idx)
        if gt_shape is None:
            gt_shape = _fallback_shape_from_camera(cameras_gt[gt_idx])
        if pre_shape is None or gt_shape is None:
            logger.info("  %s: pre_shape=%s gt_shape=%s scale=(n/a)", pre_name, pre_shape, gt_shape)
            continue
        pre_h, pre_w = pre_shape
        gt_h, gt_w = gt_shape
        if pre_h <= 0 or pre_w <= 0 or gt_h <= 0 or gt_w <= 0:
            logger.info("  %s: pre_shape=%s gt_shape=%s scale=(n/a)", pre_name, pre_shape, gt_shape)
            continue
        scale_u = gt_w / pre_w
        scale_v = gt_h / pre_h
        logger.info(
            "  %s: pre_shape=%s gt_shape=%s scale_u=%.6f scale_v=%.6f",
            pre_name,
            pre_shape,
            gt_shape,
            scale_u,
            scale_v,
        )


def _map_gt_cameras_to_preba(
    pre_ba_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: list[str],
) -> dict[int, gtsfm_types.CAMERA_TYPE]:
    name_to_gt_idx = {name: idx for idx, name in enumerate(gt_filenames)}
    for name, idx in list(name_to_gt_idx.items()):
        name_to_gt_idx[Path(name).name] = idx
    gt_by_preba_idx: dict[int, gtsfm_types.CAMERA_TYPE] = {}
    for pre_idx in pre_ba_data.get_valid_camera_indices():
        pre_name = pre_ba_data.get_image_info(pre_idx).name
        if pre_name is None:
            continue
        gt_idx = name_to_gt_idx.get(pre_name) or name_to_gt_idx.get(Path(pre_name).name)
        if gt_idx is None or gt_idx >= len(cameras_gt):
            continue
        gt_cam = cameras_gt[gt_idx]
        if gt_cam is None:
            continue
        gt_by_preba_idx[pre_idx] = gt_cam
    return gt_by_preba_idx


def _filter_tracks_with_cameras(
    pre_ba_data: GtsfmData,
    cameras_by_idx: dict[int, gtsfm_types.CAMERA_TYPE],
    reproj_err_thresh: float,
) -> tuple[GtsfmData, list[int]]:
    filtered = GtsfmData(pre_ba_data.number_images())
    filtered._image_info = pre_ba_data._clone_image_info()
    valid_track_ids = []
    for track_idx, track in enumerate(pre_ba_data.tracks()):
        if track.numberMeasurements() == 0:
            continue
        errors, _ = reprojection_utils.compute_track_reprojection_errors(cameras_by_idx, track)
        new_track = SfmTrack(track.point3())
        track_cameras = set()
        for k in range(track.numberMeasurements()):
            if np.isnan(errors[k]) or errors[k] > reproj_err_thresh:
                continue
            i, uv = track.measurement(k)
            new_track.addMeasurement(i, uv)
            track_cameras.add(i)
        if len(track_cameras) < 2:
            continue
        filtered.add_track(new_track)
        valid_track_ids.append(track_idx)
        for i in track_cameras:
            camera_i = pre_ba_data.get_camera(i)
            assert camera_i is not None
            filtered.add_camera(i, camera_i)
    return filtered, valid_track_ids


def _copy_data_with_track_ids(data: GtsfmData, track_ids: list[int]) -> GtsfmData:
    copied = GtsfmData(data.number_images())
    copied._image_info = data._clone_image_info()
    for track_idx in track_ids:
        track = data.get_track(track_idx)
        copied.add_track(track)
    for cam_idx in data.get_valid_camera_indices():
        camera = data.get_camera(cam_idx)
        assert camera is not None
        copied.add_camera(cam_idx, camera)
    return copied


def _retriangulate_tracks(pre_ba_data: GtsfmData) -> GtsfmData:
    cameras = pre_ba_data.cameras()
    point3d_initializer = Point3dInitializer(
        track_camera_dict=cameras,
        options=TriangulationOptions(mode=TriangulationSamplingMode.NO_RANSAC),
    )
    retriangulated = GtsfmData(pre_ba_data.number_images())
    retriangulated._image_info = pre_ba_data._clone_image_info()
    for cam_idx, cam in cameras.items():
        retriangulated.add_camera(cam_idx, cam)

    for track in pre_ba_data.tracks():
        if track.numberMeasurements() < 2:
            continue
        measurements: list[SfmMeasurement] = []
        for k in range(track.numberMeasurements()):
            i, uv = track.measurement(k)
            measurements.append(SfmMeasurement(i, uv))
        track_2d = SfmTrack2d(measurements)
        track_3d, _, _ = point3d_initializer.triangulate(track_2d)
        if track_3d is None:
            continue
        retriangulated.add_track(track_3d)
    return retriangulated


def _remap_tracks_to_gt(
    pre_ba_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: list[str],
    gt_shapes: Optional[list[Optional[tuple[int, int]]]],
) -> list[SfmTrack2d]:
    name_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(gt_filenames)}
    for name, idx in list(name_to_idx.items()):
        name_to_idx[Path(name).name] = idx

    pre_shapes: dict[int, Optional[tuple[int, int]]] = {}
    for cam_idx in pre_ba_data.get_valid_camera_indices():
        pre_shapes[cam_idx] = _resolve_shape(pre_ba_data.get_image_info(cam_idx).shape, pre_ba_data.get_camera(cam_idx))
    gt_shapes_by_idx: dict[int, Optional[tuple[int, int]]] = {}
    if gt_shapes is not None:
        gt_shapes_by_idx = {idx: _normalize_shape(shape) for idx, shape in enumerate(gt_shapes)}

    remapped_tracks: list[SfmTrack2d] = []
    ratio_warned: set[tuple[int, int]] = set()
    for track in pre_ba_data.tracks():
        measurements: list[SfmMeasurement] = []
        used_cams: set[int] = set()
        for k in range(track.numberMeasurements()):
            pre_idx, uv = track.measurement(k)
            pre_name = pre_ba_data.get_image_info(pre_idx).name
            if pre_name is None:
                continue
            gt_idx = name_to_idx.get(pre_name) or name_to_idx.get(Path(pre_name).name)
            if gt_idx is None or gt_idx >= len(cameras_gt):
                continue
            if cameras_gt[gt_idx] is None:
                continue
            if gt_idx in used_cams:
                continue
            pre_shape = pre_shapes.get(pre_idx)
            gt_shape = gt_shapes_by_idx.get(gt_idx)
            if gt_shape is None:
                gt_shape = _fallback_shape_from_camera(cameras_gt[gt_idx])
            if pre_shape is None or gt_shape is None:
                continue
            pre_h, pre_w = pre_shape
            gt_h, gt_w = gt_shape
            if pre_h <= 0 or pre_w <= 0 or gt_h <= 0 or gt_w <= 0:
                continue
            scale_u = gt_w / pre_w
            scale_v = gt_h / pre_h
            if abs(scale_u - scale_v) > 1e-3 * max(scale_u, scale_v):
                pair_key = (pre_idx, gt_idx)
                if pair_key not in ratio_warned:
                    ratio_warned.add(pair_key)
                    logger.warning(
                        "Aspect ratio mismatch for %s -> %s (pre HxW %s, gt HxW %s).",
                        pre_ba_data.get_image_info(pre_idx).name,
                        gt_filenames[gt_idx],
                        pre_shape,
                        gt_shape,
                    )
            scaled_uv = np.array([uv[0] * scale_u, uv[1] * scale_v], dtype=float)
            measurements.append(SfmMeasurement(gt_idx, scaled_uv))
            used_cams.add(gt_idx)
        if len(measurements) >= 2:
            remapped_tracks.append(SfmTrack2d(measurements))
    return remapped_tracks


def _triangulate_tracks(
    cameras: dict[int, gtsfm_types.CAMERA_TYPE],
    tracks_2d: list[SfmTrack2d],
) -> list[SfmTrack]:
    point3d_initializer = Point3dInitializer(
        track_camera_dict=cameras,
        options=TriangulationOptions(
            mode=TriangulationSamplingMode.NO_RANSAC,
            # reproj_error_threshold=triangulation_reproj_error_thresh,
            # min_triangulation_angle=min_triangulation_angle,
        ),
    )
    triangulated_tracks: list[SfmTrack] = []
    for track_2d in tracks_2d:
        track_3d, _, _ = point3d_initializer.triangulate(track_2d)
        if track_3d is None:
            continue
        triangulated_tracks.append(track_3d)
    return triangulated_tracks


def _save_reprojection_error_histograms(
    initial_data: GtsfmData,
    post_ba_data: GtsfmData,
    output_dir: Path,
) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore
        import plotly.io as pio  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
    except Exception:
        logger.warning("Plotly not available, skipping reprojection error histograms.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    def _errors_for_tracks(cameras: dict[int, gtsfm_types.CAMERA_TYPE], tracks_source: list[SfmTrack]) -> np.ndarray:
        errors_list: list[np.ndarray] = []
        for track in tracks_source:
            if track.numberMeasurements() == 0:
                continue
            errors, _ = reprojection_utils.compute_track_reprojection_errors(cameras, track)
            valid = errors[~np.isnan(errors)]
            if valid.size > 0:
                errors_list.append(valid)
        if not errors_list:
            return np.array([])
        return np.concatenate(errors_list)

    def _signed_errors_for_tracks(
        cameras: dict[int, gtsfm_types.CAMERA_TYPE],
        tracks_source: list[SfmTrack],
    ) -> tuple[np.ndarray, np.ndarray]:
        dx_list: list[np.ndarray] = []
        dy_list: list[np.ndarray] = []
        for track in tracks_source:
            if track.numberMeasurements() == 0:
                continue
            measured_uvs = []
            projected_uvs = []
            valid_mask = []
            for k in range(track.numberMeasurements()):
                cam_idx, uv_measured = track.measurement(k)
                if cam_idx not in cameras:
                    measured_uvs.append(uv_measured)
                    projected_uvs.append(np.array([np.nan, np.nan]))
                    valid_mask.append(False)
                    continue
                uv_reproj, success = cameras[cam_idx].projectSafe(track.point3())
                measured_uvs.append(uv_measured)
                projected_uvs.append(uv_reproj if success else np.array([np.nan, np.nan]))
                valid_mask.append(success)
            measured = np.array(measured_uvs)
            projected = np.array(projected_uvs)
            signed = measured - projected
            valid = np.array(valid_mask)
            if signed.size == 0:
                continue
            signed[~valid] = np.nan
            dx = signed[:, 0]
            dy = signed[:, 1]
            dx_list.append(dx[~np.isnan(dx)])
            dy_list.append(dy[~np.isnan(dy)])
        if not dx_list or not dy_list:
            return np.array([]), np.array([])
        return np.concatenate(dx_list), np.concatenate(dy_list)

    def _errors_per_camera(
        cameras: dict[int, gtsfm_types.CAMERA_TYPE], tracks_source: list[SfmTrack]
    ) -> dict[int, np.ndarray]:
        per_camera: dict[int, list[float]] = {}
        for track in tracks_source:
            if track.numberMeasurements() == 0:
                continue
            errors, _ = reprojection_utils.compute_track_reprojection_errors(cameras, track)
            for k in range(track.numberMeasurements()):
                cam_idx, _ = track.measurement(k)
                if cam_idx not in cameras:
                    continue
                err = errors[k]
                if np.isnan(err):
                    continue
                per_camera.setdefault(cam_idx, []).append(float(err))
        return {cam_idx: np.array(vals, dtype=float) for cam_idx, vals in per_camera.items()}

    def _mean_errors_for_tracks(
        cameras: dict[int, gtsfm_types.CAMERA_TYPE], tracks_source: list[SfmTrack]
    ) -> np.ndarray:
        means: list[float] = []
        for track in tracks_source:
            if track.numberMeasurements() == 0:
                continue
            errors, _ = reprojection_utils.compute_track_reprojection_errors(cameras, track)
            valid = errors[~np.isnan(errors)]
            if valid.size == 0:
                continue
            means.append(float(np.mean(valid)))
        return np.array(means, dtype=float)

    init_tracks = list(initial_data.tracks())
    post_tracks = list(post_ba_data.tracks())

    label_to_errors: dict[str, np.ndarray] = {
        "gt_init": _errors_for_tracks(initial_data.cameras(), init_tracks),
        "post_ba": _errors_for_tracks(post_ba_data.cameras(), post_tracks),
    }

    def _plot_multiclass_hist(errors_by_label: dict[str, np.ndarray], filename: str) -> None:
        valid_labels = {label: errs for label, errs in errors_by_label.items() if errs.size > 0}
        if not valid_labels:
            logger.info("Skipping reprojection histogram (no errors).")
            return
        all_errors = np.concatenate(list(valid_labels.values()))
        bin_count = 80
        min_err = float(np.min(all_errors))
        max_err = min(float(np.max(all_errors)), 1000.0)
        if max_err <= min_err:
            max_err = min_err + 1.0
        bin_size = (max_err - min_err) / bin_count
        fig = go.Figure()
        for label, errors in valid_labels.items():
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    xbins=dict(start=min_err, end=max_err, size=bin_size),
                    name=label,
                    opacity=0.6,
                )
            )
        fig.update_layout(
            title="Reprojection error distributions",
            xaxis_title="Reprojection error (px)",
            yaxis_title="Track count",
            barmode="overlay",
            bargap=0.05,
        )
        html_path = output_dir / filename
        pio.write_html(fig, file=str(html_path), auto_open=False)

    def _plot_nll_curve(errors_by_label: dict[str, np.ndarray], filename: str, bins: int = 100) -> None:
        valid_labels = {label: errs for label, errs in errors_by_label.items() if errs.size > 0}
        if not valid_labels:
            logger.info("Skipping NLL plot (no errors).")
            return
        fig = go.Figure()
        for label, errors in valid_labels.items():
            hist, edges = np.histogram(errors, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            nll = -np.log(hist + 1e-12)
            fig.add_trace(go.Scatter(x=centers, y=nll, mode="lines", name=label))
        fig.update_layout(
            title="Signed reprojection error NLL",
            xaxis_title="Signed reprojection error (px)",
            yaxis_title="Negative log-likelihood",
        )
        html_path = output_dir / filename
        pio.write_html(fig, file=str(html_path), auto_open=False)

    _plot_multiclass_hist(label_to_errors, "reprojection_error_histograms.html")

    signed_errors_x: dict[str, np.ndarray] = {}
    signed_errors_y: dict[str, np.ndarray] = {}
    for label, cameras, tracks_source in (
        ("gt_init", initial_data.cameras(), init_tracks),
        ("post_ba", post_ba_data.cameras(), post_tracks),
    ):
        dx, dy = _signed_errors_for_tracks(cameras, tracks_source)
        signed_errors_x[label] = dx
        signed_errors_y[label] = dy
    _plot_multiclass_hist(signed_errors_x, "reprojection_error_signed_x.html")
    _plot_multiclass_hist(signed_errors_y, "reprojection_error_signed_y.html")
    _plot_nll_curve(signed_errors_x, "reprojection_error_signed_x_nll.html")
    _plot_nll_curve(signed_errors_y, "reprojection_error_signed_y_nll.html")

    track_mean_errors = {
        "gt_init": _mean_errors_for_tracks(initial_data.cameras(), init_tracks),
        "post_ba": _mean_errors_for_tracks(post_ba_data.cameras(), post_tracks),
    }
    _plot_multiclass_hist(track_mean_errors, "reprojection_error_track_means.html")

    def _plot_per_camera_histograms_multiclass(
        cameras_by_label: dict[str, dict[int, gtsfm_types.CAMERA_TYPE]],
        tracks_by_label: dict[str, list[SfmTrack]],
        filename: str,
    ) -> None:
        per_camera_by_label = {
            label: _errors_per_camera(cams, tracks_by_label[label]) for label, cams in cameras_by_label.items()
        }
        cam_indices = sorted({cam_idx for data in per_camera_by_label.values() for cam_idx in data.keys()})
        if not cam_indices:
            logger.info("Skipping per-camera reprojection histograms (no errors).")
            return
        n = len(cam_indices)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"cam {i}" for i in cam_indices])
        for idx, cam_idx in enumerate(cam_indices):
            row = idx // cols + 1
            col = idx % cols + 1
            combined = []
            for label in per_camera_by_label:
                errors = per_camera_by_label[label].get(cam_idx)
                if errors is not None and errors.size > 0:
                    combined.append(errors)
            if not combined:
                continue
            all_errors = np.concatenate(combined)
            bin_count = 40
            min_err = float(np.min(all_errors))
            max_err = min(float(np.max(all_errors)), 1000.0)
            if max_err <= min_err:
                max_err = min_err + 1.0
            bin_size = (max_err - min_err) / bin_count
            for label, data in per_camera_by_label.items():
                errors = data.get(cam_idx)
                if errors is None or errors.size == 0:
                    continue
                fig.add_trace(
                    go.Histogram(
                        x=errors,
                        xbins=dict(start=min_err, end=max_err, size=bin_size),
                        name=label,
                        opacity=0.6,
                        legendgroup=label,
                        showlegend=(idx == 0),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_layout(
            title="Per-camera reprojection errors",
            barmode="overlay",
            height=250 * rows,
            width=300 * cols,
        )
        html_path = output_dir / filename
        pio.write_html(fig, file=str(html_path), auto_open=False)

    cameras_by_label = {
        "gt_init": initial_data.cameras(),
        "post_ba": post_ba_data.cameras(),
    }
    tracks_by_label = {
        "gt_init": init_tracks,
        "post_ba": post_tracks,
    }
    _plot_per_camera_histograms_multiclass(
        cameras_by_label,
        tracks_by_label,
        "reprojection_error_per_camera.html",
    )


def _update_cameras_with_gt(
    pre_ba_data: GtsfmData,
    pre_ba_to_gt_map: dict[int, int],
    gt_cameras: list[Optional[gtsfm_types.CAMERA_TYPE]],
) -> GtsfmData:
    updated = GtsfmData(pre_ba_data.number_images())
    updated._image_info = pre_ba_data._clone_image_info()
    for cam_idx in pre_ba_data.get_valid_camera_indices():
        new_cam_idx = pre_ba_to_gt_map[cam_idx]
        camera = gt_cameras[new_cam_idx]
        assert camera is not None
        updated.add_camera(cam_idx, camera)
    return updated


def run_bundle_adjustment_from_gt(
    pre_ba_dir: str,
    output_dir: str,
    print_summary: bool,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: list[str],
    gt_shapes: Optional[list[Optional[tuple[int, int]]]],
    loader: ColmapLoader | OlssonLoader,
    pre_ba_filter_reproj_thresh: Optional[float],
    min_viz_error_px: float,
    save_track_viz: bool,
    use_preba_intrinsics: bool,
    factor_graph_output_path: Optional[str],
    save_gt_filtered_preba: bool,
    gt_filter_reproj_thresh: Optional[float],
    gt_filtered_subdir: str,
) -> tuple[GtsfmData, GtsfmData]:
    pre_ba_data = _load_pre_ba_data(pre_ba_dir)
    if pre_ba_filter_reproj_thresh is not None:
        filtered_pre_ba_data = pre_ba_data.filter_landmark_measurements(pre_ba_filter_reproj_thresh)
    else:
        filtered_pre_ba_data = pre_ba_data
    cameras_gt, gt_filenames, gt_shapes = _filter_gt_to_preba(filtered_pre_ba_data, cameras_gt, gt_filenames, gt_shapes)
    if use_preba_intrinsics:
        _log_intrinsics_comparison(filtered_pre_ba_data, cameras_gt, gt_filenames)
        cameras_gt = _override_gt_intrinsics_with_preba(filtered_pre_ba_data, cameras_gt, gt_filenames)
    _log_image_scales(filtered_pre_ba_data, cameras_gt, gt_filenames, gt_shapes)

    if save_gt_filtered_preba:
        if gt_filter_reproj_thresh is None:
            raise ValueError("--gt_filter_reproj_thresh is required when --save_gt_filtered_preba is set.")
        gt_cameras_preba = _map_gt_cameras_to_preba(filtered_pre_ba_data, cameras_gt, gt_filenames)
        # Update the pre-BA data with the gt cameras
        new_data = GtsfmData(filtered_pre_ba_data.number_images())
        new_data._image_info = filtered_pre_ba_data._clone_image_info()
        for cam_idx in filtered_pre_ba_data.get_valid_camera_indices():
            camera = gt_cameras_preba[cam_idx]
            assert camera is not None
            new_data.add_camera(cam_idx, camera)
        for track_idx in range(filtered_pre_ba_data.number_tracks()):
            track = filtered_pre_ba_data.get_track(track_idx)
            new_data.add_track(track)

        retriangulated_preba = _retriangulate_tracks(new_data)
        _, valid_track_ids = _filter_tracks_with_cameras(
            pre_ba_data=retriangulated_preba,
            cameras_by_idx=gt_cameras_preba,
            reproj_err_thresh=gt_filter_reproj_thresh,
        )
        retriangulated_preba = _copy_data_with_track_ids(retriangulated_preba, valid_track_ids)
        gt_filtered_dir = Path(pre_ba_dir).parent / gt_filtered_subdir
        retriangulated_preba.export_as_colmap_text(str(gt_filtered_dir))
        logger.info("Saved GT-filtered pre-BA model to %s with %d tracks", gt_filtered_dir, len(valid_track_ids))

    tracks_2d = _remap_tracks_to_gt(
        filtered_pre_ba_data,
        cameras_gt,
        gt_filenames,
        gt_shapes,
    )
    cameras_gt_dict = {idx: cam for idx, cam in enumerate(cameras_gt) if cam is not None}
    triangulated_tracks = _triangulate_tracks(
        cameras_gt_dict,
        tracks_2d,
    )

    gt_init_data = _build_gt_init_data(cameras_gt, gt_filenames, gt_shapes, triangulated_tracks)
    logger.info(
        "Transferred %d tracks -> triangulated %d tracks",
        len(tracks_2d),
        len(triangulated_tracks),
    )

    viz_root = Path(output_dir) / "tracks_viz"
    if save_track_viz:
        visualize_reprojection_overlays(
            gt_init_data,
            loader,
            str(viz_root / "gt_init"),
            draw_measured=True,
            dot_on_measured=False,
            line_only=False,
            min_error_px=min_viz_error_px,
        )

    ba = BundleAdjustmentOptimizer(
        print_summary=print_summary,
        save_iteration_visualization=True,
        calibration_prior_noise_sigma=10.0,
        measurement_noise_sigma=2.0,
        robust_measurement_noise=True,
        shared_calib=True,
    )
    if factor_graph_output_path is not None:
        Path(factor_graph_output_path).parent.mkdir(parents=True, exist_ok=True)
    post_ba_data, final_error = ba.run_simple_ba(
        gt_init_data, verbose=True, factor_graph_output_path=factor_graph_output_path
    )
    logger.info("Final BA error: %.3f", final_error)
    logger.info("Reprojection error stats (GT init data):")
    gt_init_data.log_scene_reprojection_error_stats()
    logger.info("Reprojection error stats (post-BA result):")
    post_ba_data.log_scene_reprojection_error_stats()

    _save_post_ba_data(post_ba_data, output_dir)
    logger.info("Saved post-BA model to %s", output_dir)

    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_init = _get_pose_metrics(gt_init_data, cameras_gt, gt_filenames, save_dir=str(metrics_dir))
    metrics_post = _get_pose_metrics(post_ba_data, cameras_gt, gt_filenames, save_dir=str(metrics_dir))
    metrics_post.add_metric(GtsfmMetric("final_ba_error", final_error))
    metrics_init_path = metrics_dir / "ba_metrics_gt_init.json"
    metrics_post_path = metrics_dir / "ba_metrics_post_ba.json"
    metrics_init.save_to_json(str(metrics_init_path))
    metrics_post.save_to_json(str(metrics_post_path))
    logger.info("Saved GT-init BA metrics to %s", metrics_init_path)
    logger.info("Saved post-BA metrics to %s", metrics_post_path)

    _save_reprojection_error_histograms(
        initial_data=gt_init_data,
        post_ba_data=post_ba_data,
        output_dir=metrics_dir / "plots",
    )

    if save_track_viz:
        visualize_reprojection_overlays(
            post_ba_data,
            loader,
            str(viz_root / "post_ba"),
            draw_measured=True,
            dot_on_measured=False,
            line_only=True,
            min_error_px=min_viz_error_px,
        )

    return gt_init_data, post_ba_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre_ba_dir", help="Path to COLMAP text/bin model directory.")
    parser.add_argument(
        "--results_root",
        help="Root directory to recursively search for subdirs named 'vggt_pre_ba'.",
    )
    parser.add_argument(
        "--output_subdir",
        required=True,
        help="Output subdirectory name (sibling of each vggt_pre_ba folder).",
    )
    parser.add_argument("--dataset_dir", required=True, help="Dataset root for loading GT cameras via a loader.")
    parser.add_argument("--print_summary", action="store_true", help="Print GTSAM optimizer summary.")
    parser.add_argument(
        "--loader",
        choices=LOADER_CHOICES,
        default="colmap",
        help="Dataset loader type for GT cameras.",
    )
    parser.add_argument("--images_dir", default=None, help="Optional images directory for the loader.")
    parser.add_argument(
        "--no_use_gt_intrinsics",
        action="store_true",
        help="Disable GT intrinsics in the loader (default: use GT if available).",
    )
    parser.add_argument(
        "--no_use_gt_extrinsics",
        action="store_true",
        help="Disable GT extrinsics in the loader (default: use GT if available).",
    )
    parser.add_argument(
        "--pre_ba_filter_reproj_thresh",
        type=float,
        default=10.0,
        help="Optional pre-BA track filtering reprojection threshold in pixels (default: 10).",
    )
    parser.add_argument(
        "--no_pre_ba_filter",
        action="store_true",
        help="Disable pre-BA track filtering.",
    )
    parser.add_argument(
        "--min_viz_error_px",
        type=float,
        default=0.0,
        help="Only draw track points above this reprojection error (default: 0).",
    )
    parser.add_argument(
        "--save_track_viz",
        action="store_true",
        help="Save per-image reprojection visualizations to <output_dir>/tracks_viz.",
    )
    parser.add_argument(
        "--use_preba_intrinsics",
        action="store_true",
        help="Use pre-BA intrinsics with GT poses instead of GT intrinsics.",
    )
    parser.add_argument(
        "--save_factor_graph",
        action="store_true",
        help="Save the GTSAM factor graph before BA to <output_dir>/metrics/factor_graph.txt.",
    )
    parser.add_argument(
        "--save_gt_filtered_preba",
        action="store_true",
        help="Save pre-BA tracks filtered by GT reprojection error to a sibling directory.",
    )
    parser.add_argument(
        "--gt_filter_reproj_thresh",
        type=float,
        default=None,
        help="GT reprojection error threshold for filtering pre-BA tracks.",
    )
    parser.add_argument(
        "--gt_filtered_subdir",
        default="vggt_pre_ba_gt_filtered",
        help="Output subdir name for GT-filtered pre-BA data.",
    )
    parser.add_argument("--max_resolution", type=int, default=760, help="Max image short side in pixels.")
    parser.add_argument("--max_frame_lookahead", type=int, default=20, help="Olsson loader frame lookahead.")

    args = parser.parse_args()

    if args.pre_ba_dir is None and args.results_root is None:
        raise ValueError("Provide either --pre_ba_dir or --results_root.")

    loader = _create_loader(
        loader_type=args.loader,
        dataset_dir=args.dataset_dir,
        images_dir=args.images_dir,
        use_gt_intrinsics=not args.no_use_gt_intrinsics,
        use_gt_extrinsics=not args.no_use_gt_extrinsics,
        max_resolution=args.max_resolution,
        max_frame_lookahead=args.max_frame_lookahead,
    )
    gt_cameras = loader.get_gt_cameras()
    gt_filenames = loader.image_filenames()
    gt_shapes: list[Optional[tuple[int, int]]] = [shape for shape in loader.get_image_shapes()]
    num_gt = sum(camera is not None for camera in gt_cameras)
    logger.info("Loaded %d GT cameras using %s loader.", num_gt, args.loader)

    def _run_on_preba_dir(pre_ba_dir: Path) -> None:
        output_dir = pre_ba_dir.parent / args.output_subdir
        factor_graph_output_path = None
        if args.save_factor_graph:
            factor_graph_output_path = str(Path(output_dir) / "metrics" / "factor_graph.txt")
        run_bundle_adjustment_from_gt(
            pre_ba_dir=str(pre_ba_dir),
            output_dir=str(output_dir),
            print_summary=args.print_summary,
            cameras_gt=gt_cameras,
            gt_filenames=gt_filenames,
            gt_shapes=gt_shapes,
            loader=loader,
            pre_ba_filter_reproj_thresh=None if args.no_pre_ba_filter else args.pre_ba_filter_reproj_thresh,
            min_viz_error_px=args.min_viz_error_px,
            save_track_viz=args.save_track_viz,
            use_preba_intrinsics=args.use_preba_intrinsics,
            factor_graph_output_path=factor_graph_output_path,
            save_gt_filtered_preba=args.save_gt_filtered_preba,
            gt_filter_reproj_thresh=args.gt_filter_reproj_thresh,
            gt_filtered_subdir=args.gt_filtered_subdir,
        )

    if args.pre_ba_dir is not None:
        _run_on_preba_dir(Path(args.pre_ba_dir))
    if args.results_root is not None:
        results_root = Path(args.results_root)
        vggt_dirs = [p for p in results_root.rglob("vggt_pre_ba") if p.is_dir()]
        if not vggt_dirs:
            logger.warning("No 'vggt_pre_ba' directories found under %s", results_root)
        for pre_ba_dir in sorted(vggt_dirs):
            logger.info("Running BA for %s", pre_ba_dir)
            _run_on_preba_dir(pre_ba_dir)


if __name__ == "__main__":
    main()
