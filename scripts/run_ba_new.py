"""Run bundle adjustment on a pre-BA COLMAP model and save results.

Example:
    python scripts/run_ba.py \
        --pre_ba_dir /path/to/vggt_pre_ba \
        --output_dir /path/to/vggt_post_ba \
        --dataset_dir ../data/gerrard-hall \
        --loader colmap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, cast

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer, RobustBAMode
from gtsfm.bundle.bundle_adjustment_with_hooks import BundleAdjustmentWithHooks
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.loader.olsson_loader import OlssonLoader
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.geometry_comparisons as comp_utils
from gtsam import Unit3, SfmTrack

import gtsfm.utils.metrics as metrics_utils
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsam import Pose3
from gtsfm.common import types as gtsfm_types
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationOptions,
    TriangulationSamplingMode,
)
from gtsfm.utils import reprojection as reprojection_utils
import numpy as np
import gtsfm.utils.align as align_utils
import gtsam
from gtsfm.utils.ba_debug_utils import build_ba_debug_hooks


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
    colmap_files_subdir: Optional[str] = None,
) -> ColmapLoader | OlssonLoader:
    if loader_type == "colmap":
        return ColmapLoader(
            dataset_dir=dataset_dir,
            images_dir=images_dir,
            use_gt_intrinsics=use_gt_intrinsics,
            use_gt_extrinsics=use_gt_extrinsics,
            max_resolution=max_resolution,
            colmap_files_subdir=colmap_files_subdir,
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


def _get_v3d_poses(poses):
    import numpy as np
    import visu3d as v3d

    v3_poses = [
        np.concatenate([0.1 * p.rotation().matrix(), p.translation()[:, None]], axis=-1) for p in poses if p is not None
    ]
    if not v3_poses:
        return None
    v3_poses = [np.concatenate([p, np.array([[0, 0, 0, 1]], dtype=np.float64)], axis=0) for p in v3_poses]
    v3_poses = np.stack(v3_poses).astype(np.float32)
    return v3d.Transform.from_matrix(cast(Any, v3_poses))


def _pose_to_v3d_matrix(pose):
    import numpy as np

    matrix = np.concatenate([0.1 * pose.rotation().matrix(), pose.translation()[:, None]], axis=-1)
    matrix = np.concatenate([matrix, np.array([[0, 0, 0, 1]], dtype=np.float64)], axis=0)
    return matrix.astype(np.float32)


def _match_gt_poses_by_filename(
    result_data: GtsfmData,
    cameras_gt: Optional[list[Optional[gtsfm_types.CAMERA_TYPE]]],
    gt_filenames: Optional[list[str]],
) -> list[Pose3]:
    if cameras_gt is None or gt_filenames is None:
        return []
    name_to_idx = {name: idx for idx, name in enumerate(gt_filenames)}
    gt_poses = []
    for i in result_data.get_valid_camera_indices():
        name = result_data.get_image_info(i).name
        if name is None:
            continue
        gt_idx = name_to_idx.get(name)
        if gt_idx is None:
            gt_idx = name_to_idx.get(Path(name).name)
        if gt_idx is None or gt_idx >= len(cameras_gt):
            continue
        camera = cameras_gt[gt_idx]
        if camera is not None:
            gt_poses.append(camera.pose())
    return gt_poses


def _get_pose_metrics(
    result_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: Optional[list[str]] = None,
    save_dir: Optional[str] = None,
) -> GtsfmMetricsGroup:
    """Compute pose metrics for a VGGT result after aligning with ground truth."""
    poses_gt: dict[int, Pose3] = {}
    if gt_filenames:
        name_to_idx = {name: idx for idx, name in enumerate(gt_filenames)}
        for i in result_data.get_valid_camera_indices():
            name = result_data.get_image_info(i).name
            if name is None:
                continue
            gt_idx = name_to_idx.get(name)
            if gt_idx is None:
                gt_idx = name_to_idx.get(Path(name).name)
            if gt_idx is None or gt_idx >= len(cameras_gt):
                continue
            camera = cameras_gt[gt_idx]
            if camera is not None:
                poses_gt[i] = camera.pose()
    else:
        image_idxs = list(result_data._image_info.keys())
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


def _save_v3d_figures(
    pre_ba_data: GtsfmData,
    post_ba_data: GtsfmData,
    output_dir: str,
    save_html: bool,
) -> None:
    if not save_html:
        return

    try:
        import numpy as np
        import visu3d as v3d
    except ImportError as exc:
        raise ImportError("visu3d is required to save v3d figures.") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    camera_idxs = pre_ba_data.get_valid_camera_indices()
    pre_poses = []
    post_poses = []
    for i in camera_idxs:
        pre_cam = pre_ba_data.get_camera(i)
        if pre_cam is not None:
            pre_poses.append(pre_cam.pose())
        post_cam = post_ba_data.get_camera(i)
        if post_cam is not None:
            post_poses.append(post_cam.pose())

    pre_ba_p = _get_v3d_poses(pre_poses)
    post_ba_p = _get_v3d_poses(post_poses)

    pre_ba_tracks = np.array([t.point3() for t in pre_ba_data.tracks()])
    post_ba_tracks = np.array([t.point3() for t in post_ba_data.tracks()])

    fig = v3d.make_fig(
        [
            pre_ba_p,
            post_ba_p,
            v3d.Point3d(p=pre_ba_tracks),
            v3d.Point3d(p=post_ba_tracks),
        ]
    )

    if save_html:
        html_path = output_path / "ba_compare.html"
        fig.write_html(str(html_path))
        logger.info("Saved visu3d HTML to %s", html_path)

    per_camera_poses = []
    for idx, cam_idx in enumerate(camera_idxs):
        cam = post_ba_data.get_camera(cam_idx)
        if cam is None:
            continue
        v3_pose_matrix = _pose_to_v3d_matrix(cam.pose())
        per_camera_poses.append(v3d.Transform.from_matrix(cast(Any, v3_pose_matrix)))

    fig_per_camera = v3d.make_fig(
        [
            *per_camera_poses,
            v3d.Point3d(p=pre_ba_tracks),
            v3d.Point3d(p=post_ba_tracks),
        ]
    )
    html_path = output_path / "ba_compare_per_camera.html"
    fig_per_camera.write_html(str(html_path))
    logger.info("Saved per-camera visu3d HTML to %s", html_path)


def run_bundle_adjustment(
    pre_ba_dir: str,
    output_dir: str,
    print_summary: bool,
    cameras_gt: Optional[list[Optional[Any]]] = None,
    gt_filenames: Optional[list[str]] = None,
    pre_ba_filter_reproj_thresh: Optional[float] = 10.0,
    gt_tracks_data: Optional[GtsfmData] = None,
    use_gt_tracks_for_reproj: bool = False,
    run_triangulation_ba: bool = False,
    replace_intrinsics_with_gt: bool = False,
    factor_graph_output_path: Optional[str] = None,
    per_stage_basins: Optional[list[float]] = None,
) -> tuple[GtsfmData, GtsfmData, Optional[GtsfmData]]:
    pre_ba_data = _load_pre_ba_data(pre_ba_dir)
    post_ba_data = pre_ba_data
    if replace_intrinsics_with_gt and cameras_gt is not None and gt_filenames is not None:
        _replace_intrinsics_with_gt(pre_ba_data, cameras_gt, gt_filenames)
    if pre_ba_filter_reproj_thresh is not None:
        filtered_pre_ba_data = pre_ba_data.filter_landmark_measurements(pre_ba_filter_reproj_thresh)
    else:
        filtered_pre_ba_data = pre_ba_data

    pre_ba_data_optim = filtered_pre_ba_data
    # for cam_idx in pre_ba_data_optim.get_valid_camera_indices():
    #     camera = pre_ba_data_optim.get_camera(cam_idx)
    #     assert camera is not None
    #     calib = camera.calibration()
    #     f = (calib.fx() + calib.fy()) / 2.0
    #     calib = gtsam.Cal3DS2(f, f, 0.0, calib.px(), calib.py(), 0.0, 0.0, 0.0, 0.0)
    #     pre_ba_data_optim._cameras[cam_idx] = gtsam.PinholeCameraCal3DS2(camera.pose(), calib)

    if per_stage_basins is None:
        per_stage_basins = [0.8, 0.6, 0.4, 0.2]
    if not per_stage_basins:
        raise ValueError("per_stage_basins must contain at least one value.")
    ba: Optional[BundleAdjustmentOptimizer] = None
    final_error: Optional[float] = None
    for stage, basin in enumerate(per_stage_basins):
        factor_graph_filename = f"factor_graph_stage_{stage}.txt"
        if factor_graph_output_path is not None:
            factor_graph_filename = Path(factor_graph_output_path).name.replace(".txt", f"_stage_{stage}.txt")
        hooks = build_ba_debug_hooks(
            metrics_dir=Path(output_dir) / "metrics",
            save_factor_graph=factor_graph_output_path is not None,
            save_iteration_visualization=True,
            factor_graph_filename=factor_graph_filename,
        )
        ba = BundleAdjustmentWithHooks(
            print_summary=print_summary,
            save_iteration_visualization=True,
            robust_ba_mode=RobustBAMode.GMC,
            robust_noise_basin=basin,
            max_iterations=10 if stage < len(per_stage_basins) - 1 else None,
            hooks=hooks,
        )

        logger.info("Filtering retained %d/%d tracks", pre_ba_data_optim.number_tracks(), pre_ba_data.number_tracks())
        logger.info("Running BA on filtered pre-BA data stage %d", stage)
        post_ba_data, final_error = ba.run_simple_ba(pre_ba_data_optim, verbose=True)
        logger.info("Final BA error: %.3f", final_error)
        pre_ba_data_optim = post_ba_data
    if ba is None or final_error is None:
        raise RuntimeError("BA optimization did not run; check per_stage_basins.")

    _save_post_ba_data(post_ba_data, output_dir)
    logger.info("Saved post-BA model to %s", output_dir)
    logger.info("Reprojection error stats (initial data):")
    pre_ba_data.log_scene_reprojection_error_stats()
    logger.info("Reprojection error stats (filtered pre-BA data):")
    filtered_pre_ba_data.log_scene_reprojection_error_stats()
    logger.info("Reprojection error stats (post-BA result):")
    post_ba_data.log_scene_reprojection_error_stats()
    _save_reprojection_error_histograms(
        pre_ba_data=pre_ba_data,
        post_ba_data=post_ba_data,
        cameras_gt=cameras_gt,
        gt_filenames=gt_filenames,
        gt_tracks_data=gt_tracks_data,
        use_gt_tracks_for_reproj=use_gt_tracks_for_reproj,
        output_dir=Path(output_dir) / "metrics" / "plots",
    )

    triangulated_post_ba_data: Optional[GtsfmData] = None
    tri_final_error: Optional[float] = None
    if run_triangulation_ba:
        triangulated_pre_ba = _build_triangulated_pre_ba_data(pre_ba_data)
        if pre_ba_filter_reproj_thresh is not None:
            triangulated_pre_ba = triangulated_pre_ba.filter_landmark_measurements(pre_ba_filter_reproj_thresh)
        triangulated_post_ba_data, tri_final_error = ba.run_simple_ba(triangulated_pre_ba, verbose=True)
        logger.info("Final triangulation BA error: %.3f", tri_final_error)

    if cameras_gt is not None:
        metrics_dir = Path(output_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "ba_metrics.json"
        metrics_pre_ba = _get_pose_metrics(pre_ba_data, cameras_gt, gt_filenames, save_dir=str(metrics_dir))
        metrics = _get_pose_metrics(post_ba_data, cameras_gt, gt_filenames, save_dir=str(metrics_dir))
        metrics.add_metric(GtsfmMetric("final_ba_error", final_error))

        metrics_path_pre_ba = metrics_dir / "ba_metrics_pre_ba.json"
        metrics_pre_ba.save_to_json(str(metrics_path_pre_ba))

        logger.info("Saved pre-BA BA metrics to %s", metrics_path_pre_ba)
        metrics.save_to_json(str(metrics_path))
        logger.info("Saved BA metrics to %s", metrics_path)
        if triangulated_post_ba_data is not None and tri_final_error is not None:
            metrics_tri = _get_pose_metrics(
                triangulated_post_ba_data, cameras_gt, gt_filenames, save_dir=str(metrics_dir)
            )
            metrics_tri.add_metric(GtsfmMetric("final_ba_error", tri_final_error))
            metrics_path_tri = metrics_dir / "ba_metrics_triangulated.json"
            metrics_tri.save_to_json(str(metrics_path_tri))
            logger.info("Saved triangulation BA metrics to %s", metrics_path_tri)
        _log_per_camera_pose_error_changes(pre_ba_data, post_ba_data, cameras_gt)
    return pre_ba_data, post_ba_data, triangulated_post_ba_data


def _log_per_camera_pose_error_changes(
    pre_ba_data: GtsfmData, post_ba_data: GtsfmData, cameras_gt: list[Optional[Any]]
) -> None:
    input_image_idxs = pre_ba_data.get_valid_camera_indices()
    poses_gt: dict[int, Pose3] = {}
    for i in input_image_idxs:
        if i >= len(cameras_gt):
            continue
        camera = cameras_gt[i]
        if camera is not None:
            poses_gt[i] = camera.pose()
    if not poses_gt:
        logger.info("No GT poses available for per-camera error comparison.")
        return

    aligned_pre = pre_ba_data.align_via_sim3_and_transform(poses_gt)
    aligned_post = post_ba_data.align_via_sim3_and_transform(poses_gt)

    worse = []
    improved = []
    unchanged = []

    for cam_idx, gt_pose in poses_gt.items():
        pre_camera = aligned_pre.get_camera(cam_idx)
        post_camera = aligned_post.get_camera(cam_idx)
        pre_pose = pre_camera.pose() if pre_camera is not None else None
        post_pose = post_camera.pose() if post_camera is not None else None
        pre_rot = (
            comp_utils.compute_relative_rotation_angle(pre_pose.rotation(), gt_pose.rotation()) if pre_pose else None
        )
        post_rot = (
            comp_utils.compute_relative_rotation_angle(post_pose.rotation(), gt_pose.rotation()) if post_pose else None
        )
        pre_trans = (
            comp_utils.compute_relative_unit_translation_angle(
                Unit3(pre_pose.translation()), Unit3(gt_pose.translation())
            )
            if pre_pose
            else None
        )
        post_trans = (
            comp_utils.compute_relative_unit_translation_angle(
                Unit3(post_pose.translation()), Unit3(gt_pose.translation())
            )
            if post_pose
            else None
        )
        if pre_rot is None or post_rot is None or pre_trans is None or post_trans is None:
            continue
        eps = 1e-9
        rot_pct = (post_rot - pre_rot) / max(pre_rot, eps) * 100.0
        trans_pct = (post_trans - pre_trans) / max(pre_trans, eps) * 100.0
        net_pct = rot_pct + trans_pct
        if net_pct > 0.0:
            worse.append((cam_idx, pre_rot, post_rot, pre_trans, post_trans, net_pct))
        elif net_pct < 0.0:
            improved.append((cam_idx, pre_rot, post_rot, pre_trans, post_trans, net_pct))
        else:
            unchanged.append((cam_idx, pre_rot, post_rot, pre_trans, post_trans, net_pct))

    logger.info(
        "Per-camera pose error changes: worse=%d improved=%d unchanged=%d", len(worse), len(improved), len(unchanged)
    )
    if worse:
        logger.info("Cameras worse (cam_idx: pre_rot->post_rot deg, pre_trans->post_trans, net_pct):")
        for cam_idx, pre_rot, post_rot, pre_trans, post_trans, net_pct in worse:
            logger.info(
                "  %d: rot %.3f->%.3f deg, trans %.3f->%.3f, net %+0.2f%%",
                cam_idx,
                pre_rot,
                post_rot,
                pre_trans,
                post_trans,
                net_pct,
            )
    if improved:
        logger.info("Cameras improved (cam_idx: pre_rot->post_rot deg, pre_trans->post_trans, net_pct):")
        for cam_idx, pre_rot, post_rot, pre_trans, post_trans, net_pct in improved:
            logger.info(
                "  %d: rot %.3f->%.3f deg, trans %.3f->%.3f, net %+0.2f%%",
                cam_idx,
                pre_rot,
                post_rot,
                pre_trans,
                post_trans,
                net_pct,
            )


def _build_triangulated_pre_ba_data(pre_ba_data: GtsfmData) -> GtsfmData:
    cameras = pre_ba_data.cameras()
    tracks_2d: list[SfmTrack2d] = []
    for track in pre_ba_data.tracks():
        if track.numberMeasurements() < 2:
            continue
        measurements = []
        for k in range(track.numberMeasurements()):
            cam_idx, uv = track.measurement(k)
            measurements.append(SfmMeasurement(cam_idx, np.array(uv, dtype=float)))
        tracks_2d.append(SfmTrack2d(measurements))

    point3d_initializer = Point3dInitializer(
        track_camera_dict=cameras,
        options=TriangulationOptions(mode=TriangulationSamplingMode.NO_RANSAC),
    )
    triangulated_tracks: list[SfmTrack] = []
    for track_2d in tracks_2d:
        track_3d, _, _ = point3d_initializer.triangulate(track_2d)
        if track_3d is None:
            continue
        if track_3d.numberMeasurements() < 2:
            continue
        triangulated_tracks.append(track_3d)

    triangulated_data = GtsfmData.from_cameras_and_tracks(
        cameras=cameras,
        tracks=triangulated_tracks,
        number_images=pre_ba_data.number_images(),
        image_info=pre_ba_data._clone_image_info(),
        gaussian_splats=pre_ba_data.get_gaussian_splats(),
    )
    return triangulated_data


def _replace_intrinsics_with_gt(
    pre_ba_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_filenames: list[str],
) -> None:
    name_to_idx = {name: idx for idx, name in enumerate(gt_filenames)}
    for cam_idx in pre_ba_data.get_valid_camera_indices():
        name = pre_ba_data.get_image_info(cam_idx).name
        if name is None:
            continue
        gt_idx = name_to_idx.get(name) or name_to_idx.get(Path(name).name)
        if gt_idx is None or gt_idx >= len(cameras_gt):
            continue
        gt_cam = cameras_gt[gt_idx]
        if gt_cam is None:
            continue
        pre_cam = pre_ba_data.get_camera(cam_idx)
        if pre_cam is None:
            continue
        calib = gt_cam.calibration()
        camera_class = gtsfm_types.get_camera_class_for_calibration(calib)
        pre_ba_data._cameras[cam_idx] = camera_class(pre_cam.pose(), calib)  # type: ignore[attr-defined]


def _save_reprojection_error_histograms(
    pre_ba_data: GtsfmData,
    post_ba_data: GtsfmData,
    cameras_gt: Optional[list[Optional[gtsfm_types.CAMERA_TYPE]]],
    gt_filenames: Optional[list[str]],
    gt_tracks_data: Optional[GtsfmData],
    use_gt_tracks_for_reproj: bool,
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

    def _remap_gt_tracks_to_preba_indices(gt_data: GtsfmData, preba_data: GtsfmData) -> list[SfmTrack]:
        gt_idx_to_name = {}
        for idx in range(gt_data.number_images()):
            name = gt_data.get_image_info(idx).name
            if name is not None:
                gt_idx_to_name[idx] = name
        preba_name_to_idx = {}
        for idx in preba_data.get_valid_camera_indices():
            name = preba_data.get_image_info(idx).name
            if name is None:
                continue
            preba_name_to_idx[name] = idx
            preba_name_to_idx[Path(name).name] = idx
        allowed_cameras = set(preba_data.get_valid_camera_indices())
        remapped_tracks: list[SfmTrack] = []
        for track in gt_data.tracks():
            if track.numberMeasurements() == 0:
                continue
            new_track = SfmTrack(track.point3())
            new_track.r = track.r
            new_track.g = track.g
            new_track.b = track.b
            for k in range(track.numberMeasurements()):
                gt_cam_idx, uv = track.measurement(k)
                gt_name = gt_idx_to_name.get(gt_cam_idx)
                if gt_name is None:
                    continue
                preba_idx = preba_name_to_idx.get(gt_name)
                if preba_idx is None:
                    preba_idx = preba_name_to_idx.get(Path(gt_name).name)
                if preba_idx is None:
                    continue
                if preba_idx not in allowed_cameras:
                    continue
                new_track.addMeasurement(preba_idx, uv)
            if new_track.numberMeasurements() >= 2:
                remapped_tracks.append(new_track)
        return remapped_tracks

    if use_gt_tracks_for_reproj and gt_tracks_data is not None:
        tracks_source = _remap_gt_tracks_to_preba_indices(gt_tracks_data, pre_ba_data)
    else:
        tracks_source = list(post_ba_data.tracks())

    def _errors_for_tracks(cameras: dict[int, gtsfm_types.CAMERA_TYPE]) -> np.ndarray:
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

    def _errors_per_camera(cameras: dict[int, gtsfm_types.CAMERA_TYPE]) -> dict[int, np.ndarray]:
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
        cameras: dict[int, gtsfm_types.CAMERA_TYPE],
        track_indices: list[int],
    ) -> np.ndarray:
        means: list[float] = []
        for idx in track_indices:
            if idx >= len(tracks_source):
                continue
            track = tracks_source[idx]
            if track.numberMeasurements() == 0:
                continue
            errors, _ = reprojection_utils.compute_track_reprojection_errors(cameras, track)
            valid = errors[~np.isnan(errors)]
            if valid.size == 0:
                continue
            means.append(float(np.mean(valid)))
        return np.array(means, dtype=float)

    def _build_camera(pose: Pose3, calibration) -> gtsfm_types.CAMERA_TYPE:
        camera_class = gtsfm_types.get_camera_class_for_calibration(calibration)
        return camera_class(pose, calibration)  # type: ignore

    initial_cameras = pre_ba_data.cameras()
    ba_pose_cameras: dict[int, gtsfm_types.CAMERA_TYPE] = {}
    gt_calib_by_name: dict[str, gtsfm_types.CALIBRATION_TYPE] = {}
    if use_gt_tracks_for_reproj and gt_tracks_data is not None:
        for idx in gt_tracks_data.get_valid_camera_indices():
            name = gt_tracks_data.get_image_info(idx).name
            if name is None:
                continue
            cam = gt_tracks_data.get_camera(idx)
            if cam is None:
                continue
            gt_calib_by_name[Path(name).name] = cam.calibration()
            gt_calib_by_name[name] = cam.calibration()

    def _get_gt_calib_for_cam(cam_idx: int, fallback: gtsfm_types.CALIBRATION_TYPE) -> gtsfm_types.CALIBRATION_TYPE:
        name = pre_ba_data.get_image_info(cam_idx).name
        if name is None:
            return fallback
        return gt_calib_by_name.get(name, gt_calib_by_name.get(Path(name).name, fallback))

    for cam_idx, pre_cam in initial_cameras.items():
        ba_cam = post_ba_data.get_camera(cam_idx)
        if ba_cam is None:
            continue
        calib = pre_cam.calibration()
        if use_gt_tracks_for_reproj and gt_calib_by_name:
            calib = _get_gt_calib_for_cam(cam_idx, calib)
        ba_pose_cameras[cam_idx] = _build_camera(ba_cam.pose(), calib)

    label_to_errors: dict[str, np.ndarray] = {
        "initial_data": _errors_for_tracks(initial_cameras),
        "ba_poses_intrinsics": _errors_for_tracks(ba_pose_cameras),
    }

    valid_track_indices: list[int] = []
    for track_idx, track in enumerate(tracks_source):
        if track.numberMeasurements() == 0:
            continue
        errors, _ = reprojection_utils.compute_track_reprojection_errors(initial_cameras, track)
        if np.isnan(errors).any():
            continue
        valid_track_indices.append(track_idx)

    gt_pose_cameras: dict[int, gtsfm_types.CAMERA_TYPE] = {}
    ba_pose_gt_intr: dict[int, gtsfm_types.CAMERA_TYPE] = {}
    gt_pose_ba_intr: dict[int, gtsfm_types.CAMERA_TYPE] = {}
    gt_name_to_camera: dict[str, gtsfm_types.CAMERA_TYPE] = {}
    if use_gt_tracks_for_reproj and gt_tracks_data is not None:
        for idx in gt_tracks_data.get_valid_camera_indices():
            name = gt_tracks_data.get_image_info(idx).name
            if name is None:
                continue
            cam = gt_tracks_data.get_camera(idx)
            if cam is None:
                continue
            gt_name_to_camera[name] = cam
            gt_name_to_camera[Path(name).name] = cam
    elif cameras_gt is not None and gt_filenames is not None:
        for idx, name in enumerate(gt_filenames):
            if idx >= len(cameras_gt):
                continue
            cam = cameras_gt[idx]
            if cam is None:
                continue
            gt_name_to_camera[name] = cam
            gt_name_to_camera[Path(name).name] = cam

    if gt_name_to_camera:
        alignment_targets: dict[int, Pose3] = {}
        alignment_sources: dict[int, Pose3] = {}
        alignment_targets_ba: dict[int, Pose3] = {}
        alignment_sources_ba: dict[int, Pose3] = {}
        for cam_idx, pre_cam in initial_cameras.items():
            name = pre_ba_data.get_image_info(cam_idx).name
            if name is None:
                continue
            gt_cam = gt_name_to_camera.get(name) or gt_name_to_camera.get(Path(name).name)
            if gt_cam is None:
                continue
            alignment_targets[cam_idx] = pre_cam.pose()
            alignment_sources[cam_idx] = gt_cam.pose()
            gt_pose_cameras[cam_idx] = gt_cam
            ba_cam = post_ba_data.get_camera(cam_idx)
            if ba_cam is None:
                continue
            alignment_targets_ba[cam_idx] = ba_cam.pose()
            alignment_sources_ba[cam_idx] = gt_cam.pose()
            gt_pose_ba_intr[cam_idx] = _build_camera(gt_cam.pose(), ba_cam.calibration())

        if alignment_targets and alignment_sources:
            try:
                aSb_init = align_utils.sim3_from_Pose3_maps_robust(alignment_targets, alignment_sources)
                if use_gt_tracks_for_reproj:
                    # Keep GT cameras in GT frame (tracks are GT frame), align initial/BA to GT.
                    bSa_init = aSb_init.inverse()
                    initial_cameras = {
                        cam_idx: _build_camera(
                            bSa_init.transformFrom(cam.pose()),
                            _get_gt_calib_for_cam(cam_idx, cam.calibration()),
                        )
                        for cam_idx, cam in initial_cameras.items()
                    }
                    if alignment_targets_ba and alignment_sources_ba:
                        aSb_ba = align_utils.sim3_from_Pose3_maps_robust(alignment_targets_ba, alignment_sources_ba)
                        bSa_ba = aSb_ba.inverse()
                        ba_pose_cameras = {
                            cam_idx: _build_camera(
                                bSa_ba.transformFrom(cam.pose()),
                                _get_gt_calib_for_cam(cam_idx, cam.calibration()),
                            )
                            for cam_idx, cam in ba_pose_cameras.items()
                        }
                        ba_pose_gt_intr = {
                            cam_idx: _build_camera(bSa_ba.transformFrom(cam.pose()), gt_cam.calibration())
                            for cam_idx, cam in ba_pose_cameras.items()
                            if (gt_cam := gt_pose_cameras.get(cam_idx)) is not None
                        }
                else:
                    # Align GT cameras into initial frame for reprojection with non-GT tracks.
                    gt_pose_cameras = {
                        cam_idx: _build_camera(aSb_init.transformFrom(cam.pose()), cam.calibration())
                        for cam_idx, cam in gt_pose_cameras.items()
                    }
                    gt_pose_ba_intr = {
                        cam_idx: _build_camera(aSb_init.transformFrom(cam.pose()), cam.calibration())
                        for cam_idx, cam in gt_pose_ba_intr.items()
                    }
            except Exception as exc:
                logger.warning("Failed to align GT cameras for reprojection histograms: %s", exc)

        label_to_errors["gt_poses_intrinsics"] = _errors_for_tracks(gt_pose_cameras)
        label_to_errors["ba_poses_gt_intrinsics"] = _errors_for_tracks(ba_pose_gt_intr)
        label_to_errors["gt_poses_ba_intrinsics"] = _errors_for_tracks(gt_pose_ba_intr)

    def _plot_multiclass_hist(errors_by_label: dict[str, np.ndarray], filename: str) -> None:
        valid_labels = {label: errs for label, errs in errors_by_label.items() if errs.size > 0}
        if not valid_labels:
            logger.info("Skipping reprojection histogram (no errors).")
            return
        all_errors = np.concatenate(list(valid_labels.values()))
        bin_count = 80
        min_err = float(np.min(all_errors))
        max_err = min(float(np.max(all_errors)), 100.0)
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

    _plot_multiclass_hist(label_to_errors, "reprojection_error_histograms.html")

    def _plot_per_camera_histograms_multiclass(
        cameras_by_label: dict[str, dict[int, gtsfm_types.CAMERA_TYPE]],
        filename: str,
    ) -> None:
        per_camera_by_label = {label: _errors_per_camera(cams) for label, cams in cameras_by_label.items()}
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
            max_err = min(float(np.max(all_errors)), 100.0)
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
            title="Per-camera reprojection errors (all methods)",
            barmode="overlay",
            height=250 * rows,
            width=300 * cols,
        )
        html_path = output_dir / filename
        pio.write_html(fig, file=str(html_path), auto_open=False)

    if valid_track_indices:
        track_mean_errors = {
            "initial_data": _mean_errors_for_tracks(initial_cameras, valid_track_indices),
            "ba_poses_intrinsics": _mean_errors_for_tracks(ba_pose_cameras, valid_track_indices),
        }
        if cameras_gt is not None and gt_filenames is not None:
            track_mean_errors.update(
                {
                    "gt_poses_intrinsics": _mean_errors_for_tracks(gt_pose_cameras, valid_track_indices),
                    "ba_poses_gt_intrinsics": _mean_errors_for_tracks(ba_pose_gt_intr, valid_track_indices),
                    "gt_poses_ba_intrinsics": _mean_errors_for_tracks(gt_pose_ba_intr, valid_track_indices),
                }
            )
        _plot_multiclass_hist(track_mean_errors, "reprojection_error_track_means.html")

    cameras_by_label = {
        "initial_data": initial_cameras,
        "ba_poses_intrinsics": ba_pose_cameras,
    }
    if gt_pose_cameras:
        cameras_by_label["gt_poses_intrinsics"] = gt_pose_cameras
    if ba_pose_gt_intr:
        cameras_by_label["ba_poses_gt_intrinsics"] = ba_pose_gt_intr
    if gt_pose_ba_intr:
        cameras_by_label["gt_poses_ba_intrinsics"] = gt_pose_ba_intr
    _plot_per_camera_histograms_multiclass(cameras_by_label, "reprojection_error_per_camera.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre_ba_dir", help="Path to COLMAP text/bin model directory.")
    parser.add_argument(
        "--results_root",
        help="Root directory to recursively search for subdirs named 'vggt'.",
    )
    parser.add_argument(
        "--output_subdir",
        required=True,
        help="Output subdirectory name (sibling of each vggt folder).",
    )
    parser.add_argument("--print_summary", action="store_true", help="Print GTSAM optimizer summary.")

    parser.add_argument(
        "--loader",
        choices=LOADER_CHOICES,
        default="colmap",
        help="Dataset loader type, used only to load GT cameras if dataset_dir is provided.",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Optional dataset root for loading GT cameras via a loader.",
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
        default=14.0,
        help="Optional pre-BA track filtering reprojection threshold in pixels (default: 10).",
    )
    parser.add_argument(
        "--no_pre_ba_filter",
        action="store_true",
        help="Disable pre-BA track filtering.",
    )
    parser.add_argument("--max_resolution", type=int, default=760, help="Max image short side in pixels.")
    parser.add_argument("--max_frame_lookahead", type=int, default=20, help="Olsson loader frame lookahead.")
    parser.add_argument("--save_v3d_html", action="store_true", help="Save visu3d figures as HTML.")
    parser.add_argument(
        "--use_gt_tracks_for_reproj",
        action="store_true",
        help="Use GT tracks for reprojection histograms (requires loader GT).",
    )
    parser.add_argument(
        "--run_triangulation_ba",
        action="store_true",
        help="Run extra BA with triangulated track points from pre-BA data.",
    )
    parser.add_argument(
        "--save_factor_graph",
        action="store_true",
        help="Save the GTSAM factor graph before BA to <output_dir>/metrics/factor_graph.txt.",
    )
    parser.add_argument(
        "--replace_intrinsics_with_gt",
        action="store_true",
        help="Replace pre-BA intrinsics with GT intrinsics (matched by filename).",
    )
    parser.add_argument(
        "--per_stage_basins",
        type=str,
        default="0.8,0.6,0.4,0.2",
        help="Comma-separated robust noise basins per BA stage.",
    )
    parser.add_argument(
        "--v3d_output_dir",
        default=None,
        help="Output directory for visu3d exports (default: <output_dir>/visu3d).",
    )
    parser.add_argument(
        "--colmap_files_subdir",
        default=None,
        help="Subdirectory under dataset_dir where COLMAP files are located (default: None).",
    )
    args = parser.parse_args()
    if args.pre_ba_dir is None and args.results_root is None:
        raise ValueError("Provide either --pre_ba_dir or --results_root.")

    gt_cameras = None
    gt_filenames = None
    gt_tracks_data = None
    if args.dataset_dir is not None:
        loader = _create_loader(
            loader_type=args.loader,
            dataset_dir=args.dataset_dir,
            images_dir=args.images_dir,
            use_gt_intrinsics=not args.no_use_gt_intrinsics,
            use_gt_extrinsics=not args.no_use_gt_extrinsics,
            max_resolution=args.max_resolution,
            max_frame_lookahead=args.max_frame_lookahead,
            colmap_files_subdir=args.colmap_files_subdir,
        )
        gt_cameras = loader.get_gt_cameras()
        gt_filenames = loader.image_filenames()
        if args.use_gt_tracks_for_reproj and args.loader == "colmap":
            try:
                gt_tracks_data = GtsfmData.read_colmap(args.dataset_dir)
            except Exception as exc:
                logger.warning("Failed to load GT tracks from dataset_dir: %s", exc)
        num_gt = sum(camera is not None for camera in gt_cameras)
        logger.info("Loaded %d GT cameras using %s loader.", num_gt, args.loader)

    def _run_on_preba_dir(pre_ba_dir: Path) -> None:
        output_dir = pre_ba_dir.parent / args.output_subdir
        factor_graph_output_path = None
        if args.save_factor_graph:
            factor_graph_output_path = str(Path(output_dir) / "metrics" / "factor_graph.txt")
        pre_ba_data, post_ba_data, _ = run_bundle_adjustment(
            str(pre_ba_dir),
            str(output_dir),
            args.print_summary,
            cameras_gt=gt_cameras,
            gt_filenames=gt_filenames,
            pre_ba_filter_reproj_thresh=None if args.no_pre_ba_filter else args.pre_ba_filter_reproj_thresh,
            gt_tracks_data=gt_tracks_data,
            use_gt_tracks_for_reproj=args.use_gt_tracks_for_reproj,
            run_triangulation_ba=args.run_triangulation_ba,
            replace_intrinsics_with_gt=args.replace_intrinsics_with_gt,
            factor_graph_output_path=factor_graph_output_path,
            per_stage_basins=[float(v.strip()) for v in args.per_stage_basins.split(",") if v.strip()],
        )
        if args.save_v3d_html:
            v3d_output_dir = args.v3d_output_dir or str(output_dir / "visu3d")
            _save_v3d_figures(
                pre_ba_data=pre_ba_data,
                post_ba_data=post_ba_data,
                output_dir=v3d_output_dir,
                save_html=args.save_v3d_html,
            )

    if args.pre_ba_dir is not None:
        _run_on_preba_dir(Path(args.pre_ba_dir))
    if args.results_root is not None:
        results_root = Path(args.results_root)
        vggt_dirs = [p for p in results_root.rglob("vggt_pre_ba") if p.is_dir()]
        if not vggt_dirs:
            logger.warning("No 'vggt' directories found under %s", results_root)
        for pre_ba_dir in sorted(vggt_dirs):
            logger.info("Running BA for %s", pre_ba_dir)
            _run_on_preba_dir(pre_ba_dir)


if __name__ == "__main__":
    main()
